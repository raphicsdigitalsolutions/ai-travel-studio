"""Local travel assistant using LangGraph, OpenStreetMap, and Tavily."""

from __future__ import annotations

import operator
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Callable, Literal, TypedDict
from urllib.parse import quote_plus

import gradio as gr
import spaces
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, OpenAIError
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode


load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

# A deployment-level key is optional. Each visitor can also enter their own
# Tavily key in the startup dialog; the session key takes precedence.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()

OSM_USER_AGENT = "LangGraphTravelAgent/1.0 (educational local travel assistant)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = (
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
_geocode_cache: dict[str, tuple[float, float]] = {}
_nominatim_lock = threading.Lock()
_last_nominatim_request = 0.0


def _geocode(place: str) -> tuple[float, float]:
    global _last_nominatim_request

    normalized_place = place.strip()
    if not normalized_place:
        raise ValueError("Please provide a city or place name.")

    cache_key = normalized_place.casefold()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    with _nominatim_lock:
        if cache_key in _geocode_cache:
            return _geocode_cache[cache_key]

        wait_seconds = 1.0 - (time.monotonic() - _last_nominatim_request)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        response = requests.get(
            NOMINATIM_URL,
            params={"q": normalized_place, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": OSM_USER_AGENT},
            timeout=20,
        )
        _last_nominatim_request = time.monotonic()
        response.raise_for_status()
        results = response.json()

    if not results:
        raise ValueError(
            f"OpenStreetMap could not find a location for {normalized_place!r}."
        )

    coordinates = (float(results[0]["lat"]), float(results[0]["lon"]))
    _geocode_cache[cache_key] = coordinates
    return coordinates


def _query_nearby(
    place: str, tag_filters: list[str], radius_km: int = 5
) -> list[dict]:
    if not 1 <= radius_km <= 50:
        raise ValueError("radius_km must be between 1 and 50.")

    latitude, longitude = _geocode(place)
    radius_m = radius_km * 1000
    selectors = "".join(
        f"nwr(around:{radius_m},{latitude},{longitude}){tag_filter};\n"
        for tag_filter in tag_filters
    )
    query = f"[out:json][timeout:25];\n({selectors});\nout center tags;"
    last_error: requests.RequestException | None = None

    for endpoint in OVERPASS_URLS:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": OSM_USER_AGENT},
                timeout=20,
            )
            response.raise_for_status()
            return response.json().get("elements", [])
        except requests.RequestException as error:
            last_error = error

    map_url = "https://www.openstreetmap.org/search?query=" + quote_plus(place)
    raise RuntimeError(
        f"OpenStreetMap's Overpass services are unavailable ({last_error}). "
        f"Try again later or search the map directly: {map_url}"
    ) from last_error


def _format_places(elements: list[dict], city: str, limit: int = 5) -> str:
    if not 1 <= limit <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    lines = []
    for element in elements[:limit]:
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("brand") or "Unnamed place"
        center = element.get("center", {})
        latitude = element.get("lat", center.get("lat"))
        longitude = element.get("lon", center.get("lon"))
        if latitude is None or longitude is None:
            continue

        details = [
            f"{key.replace('addr:', '')}: {tags[key]}"
            for key in (
                "tourism",
                "amenity",
                "cuisine",
                "iata",
                "addr:street",
                "addr:city",
            )
            if tags.get(key)
        ]
        map_url = (
            f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}"
            f"#map=16/{latitude}/{longitude}"
        )
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- [{name}]({map_url}){suffix}")

    if not lines:
        return (
            f"No matching OpenStreetMap entries were found near {city}. "
            "Map data may be incomplete."
        )

    return (
        "OpenStreetMap results (verify details before travelling):\n"
        + "\n".join(lines)
        + "\n\nData: OpenStreetMap contributors "
        "(https://www.openstreetmap.org/copyright)."
    )


@tool
def find_airports_tool(city: str, radius_km: int = 50, max_results: int = 5) -> str:
    """Find mapped airports near a city; IATA codes may be missing from map data."""
    elements = _query_nearby(city, ['["aeroway"="aerodrome"]'], radius_km)
    return _format_places(elements, city, max_results)


@tool
def search_hotels_tool(city: str, radius_km: int = 5, max_results: int = 5) -> str:
    """Find mapped hotels and similar stays; OSM does not provide prices or availability."""
    elements = _query_nearby(
        city,
        ['["tourism"~"^(hotel|hostel|motel|guest_house|apartment|resort)$"]'],
        radius_km,
    )
    return _format_places(elements, city, max_results)


@tool
def search_local_places_tool(
    city: str,
    category: Literal["restaurant", "attraction"],
    radius_km: int = 5,
    max_results: int = 5,
) -> str:
    """Find mapped restaurants or tourist attractions near a city."""
    if category == "restaurant":
        filters = ['["amenity"~"^(restaurant|cafe|fast_food)$"]']
    else:
        filters = [
            '["tourism"~"^(attraction|museum|gallery|zoo|theme_park)$"]',
            '["historic"]',
            '["leisure"="park"]',
        ]

    elements = _query_nearby(city, filters, radius_km)
    return _format_places(elements, city, max_results)


@tool
def search_flights_tool(
    origin_city: str,
    destination_city: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
) -> str:
    """Create a flight-search link; OSM does not provide live schedules or fares."""
    if adults < 1:
        raise ValueError("adults must be at least 1.")

    query = (
        f"Flights from {origin_city} to {destination_city} on {departure_date} "
        f"for {adults} adults"
    )
    if return_date:
        query += f", returning {return_date}"

    url = "https://www.google.com/travel/flights?q=" + quote_plus(query)
    return (
        "OpenStreetMap has no live flight schedules or prices, so I cannot quote fares. "
        f"Search available flights here: {url}"
    )


@tool
def get_current_date_tool() -> str:
    """Return today's date so travel plans use current dates."""
    from datetime import date

    return date.today().isoformat()


@tool
def search_web_tool(query: str, config: RunnableConfig) -> str:
    """Search the live web for current travel information, advisories, and destination details."""
    configurable = config.get("configurable", {})
    api_key = (
        configurable.get("tavily_api_key")
        or TAVILY_API_KEY
    )
    if not isinstance(api_key, str) or not api_key.strip():
        return (
            "Live web search is not configured. Ask the user to enter a Tavily API "
            "key in the startup dialog, or use OpenStreetMap tools for mapped places."
        )

    try:
        search = TavilySearch(max_results=5, tavily_api_key=api_key.strip())
        result = search.invoke({"query": query})
        return str(result)
    except Exception as error:
        # Do not expose credentials or full request details in user-facing output.
        return (
            "Tavily web search failed. Check that the Tavily API key is valid "
            "and that the service is available, then try again. "
            f"Error type: {type(error).__name__}"
        )


tools = [
    search_web_tool,
    find_airports_tool,
    search_hotels_tool,
    search_local_places_tool,
    search_flights_tool,
    get_current_date_tool,
]
class TravelState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def call_model(
    state: TravelState, config: RunnableConfig
) -> dict[str, list[AIMessage]]:
    api_key = config.get("configurable", {}).get("openai_api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError("Enter your OpenAI API key to start a travel chat.")

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
    )
    response = model.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}


def route_after_model(state: TravelState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(TravelState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    route_after_model,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")
travel_agent = workflow.compile()

SYSTEM_PROMPT = """You are a practical travel-planning assistant.
Use the search_web_tool for current travel advisories, recent information, and
destination details. If it reports that no Tavily key is configured or search fails,
be transparent and do not pretend that a live search succeeded.
Use OpenStreetMap tools for mapped airports, hotels, restaurants, and attractions.
OSM data can be incomplete; hotel listings do not mean availability or prices are known.
OSM does not provide flight schedules or live fares: give the flight-search link and
never invent prices, routes, or availability. Ask a short follow-up if an essential
trip detail is missing. Clearly distinguish sourced facts from suggestions and include
map links."""


def _history_messages(history: list | None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in (history or [])[-20:]:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)

        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def verify_openai_api_key(
    api_key: str, tavily_api_key: str, trust_confirmed: bool
) -> tuple[str, str, str, str, dict]:
    """Verify OpenAI and keep both visitor-provided keys only in session state."""
    if not trust_confirmed:
        raise gr.Error("Confirm that you trust this app before sending your keys.")

    candidate = api_key.strip()
    if not candidate:
        raise gr.Error("Please enter your OpenAI API key.")

    tavily_candidate = tavily_api_key.strip()

    # Do not make a network request while the visitor is still in the startup
    # gate. Some hosted environments block or delay the model-list endpoint.
    # Store the key in this Gradio session and let the first chat request verify
    # it naturally. Invalid keys will then produce a clear chat error.
    return candidate, tavily_candidate, "", "", gr.update(visible=False)


def _prompt_handler(prompt: str) -> Callable[[], str]:
    return lambda: prompt


def travel_agent_chat(
    message: str,
    history: list | None = None,
    openai_api_key: str = "",
    tavily_api_key: str = "",
):
    if not openai_api_key.strip():
        yield "Enter your OpenAI API key in the secure prompt before planning a trip."
        return

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_history_messages(history))
    messages.append(HumanMessage(content=message))

    used_tools: list[str] = []
    for chunk in travel_agent.stream(
        {"messages": messages},
        {
            "recursion_limit": 15,
            "configurable": {
                "openai_api_key": openai_api_key,
                "tavily_api_key": tavily_api_key,
            },
        },
        stream_mode="updates",
    ):
        for node in chunk.values():
            for result in node.get("messages", []):
                if isinstance(result, ToolMessage):
                    if result.name not in used_tools:
                        used_tools.append(result.name)
                    yield f"Using {result.name}...\n\n"
                elif isinstance(result, AIMessage) and result.content:
                    yield result.content

    if used_tools:
        yield f"\n\nTools used: {', '.join(used_tools)}"


APP_CSS = """
/* ═══════════════════════════════════════════════════
   ROAMLY 3D — Full Immersive 3D UI System
   ═══════════════════════════════════════════════════ */
:root {
    --roam-ink: #0f1f28;
    --roam-muted: #55737a;
    --roam-teal: #0ea89a;
    --roam-teal-dark: #0b7a6e;
    --roam-mint: #d6f5eb;
    --roam-sand: #f5f2e9;
    --roam-perspective: 1200px;
    --roam-depth: 40px;
    --roam-glow: rgba(14,168,154,.55);
    --roam-card-bg: rgba(255,255,255,.92);
    --neon-teal: #0ef5d8;
}

.roam-status {
    width: fit-content;
    margin: 8px 4px 14px !important;
    padding: 9px 14px !important;
    border: 1px solid rgba(14,168,154,.22) !important;
    border-radius: 999px !important;
    color: #28665f !important;
    background: rgba(255,255,255,.62) !important;
    box-shadow: 0 7px 18px rgba(12,87,77,.07);
    font-size: 12px !important;
}
body.dark .roam-status {
    color: #b8eee0 !important;
    background: rgba(16,48,46,.72) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(14,168,154,.35); border-radius: 99px; }

/* ── Base ── */
body, .gradio-container {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(14,168,154,.18), transparent 36rem),
        radial-gradient(ellipse at 92% 55%, rgba(11,122,110,.14), transparent 34rem),
        radial-gradient(ellipse at 55% 95%, rgba(10,80,72,.1), transparent 28rem),
        linear-gradient(165deg, #e8f5f1 0%, #eef7f3 40%, #e4f2ee 100%) !important;
}

html, body, gradio-app { width: 100%; min-width: 0; background: #e8f4ef; }

.roam-key-gate {
    /* Do not set display here: Gradio controls visibility on this component. */
    position: fixed !important;
    z-index: 9999 !important;
    inset: 0 !important;
    padding: 20px !important;
    background: rgba(3,18,26,.76) !important;
    backdrop-filter: blur(16px) saturate(1.2);
    animation: roam-key-fade .35s ease both;
}

.roam-key-card {
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
}

.roam-key-card {
    position: relative !important;
    width: min(460px, 100%) !important;
    max-width: 460px !important;
    padding: 30px !important;
    overflow: hidden;
    border: 1px solid rgba(206,255,236,.68) !important;
    border-radius: 28px !important;
    background:
        radial-gradient(ellipse at 86% 0%, rgba(93,220,174,.22), transparent 16rem),
        linear-gradient(150deg, rgba(255,255,255,.98), rgba(233,250,242,.98)) !important;
    box-shadow:
        0 2px 0 rgba(255,255,255,.95) inset,
        0 -3px 0 rgba(9,109,82,.08) inset,
        0 36px 100px rgba(0,0,0,.4),
        0 0 0 1px rgba(96,255,204,.12),
        0 0 50px rgba(73,228,175,.15);
    transform-style: preserve-3d;
    animation: roam-key-arrive .55s cubic-bezier(.16,.8,.25,1) both;
}

.roam-key-card:before {
    position: absolute;
    top: -92px;
    right: -78px;
    width: 200px;
    height: 200px;
    border: 1px solid rgba(20,163,125,.13);
    border-radius: 50%;
    box-shadow: 0 0 0 20px rgba(20,163,125,.035), 0 0 0 44px rgba(20,163,125,.025);
    content: "";
    pointer-events: none;
}

.roam-key-icon {
    position: relative;
    display: grid;
    width: 54px;
    height: 54px;
    place-items: center;
    margin-bottom: 20px;
    border: 1px solid rgba(202,255,236,.65);
    border-radius: 17px;
    background: linear-gradient(145deg, #1ac59b, #087962 75%, #075747);
    box-shadow: 0 7px 0 rgba(5,93,70,.22), 0 14px 26px rgba(11,155,116,.27), inset 0 1px rgba(255,255,255,.42);
    transform: translateZ(18px) rotate(-5deg);
}

.roam-key-icon span {
    position: relative;
    width: 18px;
    height: 15px;
    margin-top: 8px;
    border: 2px solid #eafff5;
    border-radius: 4px;
    background: linear-gradient(145deg, #dcfff0, #82eac3);
    box-shadow: 0 2px 5px rgba(0,36,26,.24);
}

.roam-key-icon span:before {
    position: absolute;
    left: 3px;
    bottom: 11px;
    width: 8px;
    height: 10px;
    border: 2px solid #eafff5;
    border-bottom: 0;
    border-radius: 8px 8px 0 0;
    content: "";
}

.roam-key-card h2 {
    position: relative;
    margin: 0 0 10px;
    color: #102b31;
    font-size: clamp(24px, 6vw, 30px);
    font-weight: 850;
    letter-spacing: -.045em;
    transform: translateZ(10px);
}

.roam-key-copy {
    position: relative;
    margin: 0 0 19px;
    color: #567177;
    font-size: 13px;
    line-height: 1.65;
}

.roam-key-card label {
    color: #23464b !important;
    font-size: 12px !important;
    font-weight: 750 !important;
}

.roam-key-card input[type="password"] {
    min-height: 50px !important;
    border: 1px solid rgba(12,134,103,.24) !important;
    border-radius: 14px !important;
    background: linear-gradient(180deg, #eaf7f1, #fff) !important;
    box-shadow: 0 3px 9px rgba(11,91,70,.09) inset, 0 1px 0 #fff !important;
}

.roam-key-card input[type="password"]:focus {
    border-color: rgba(10,160,119,.62) !important;
    box-shadow: 0 3px 9px rgba(11,91,70,.08) inset, 0 0 0 3px rgba(16,190,139,.14) !important;
}

.roam-key-trust {
    color: #355a5d !important;
    font-size: 11px !important;
    font-weight: 650 !important;
}

.roam-key-trust input[type="checkbox"] { accent-color: #0b9e78; }

.roam-key-submit {
    min-height: 49px !important;
    margin-top: 8px !important;
    border: 0 !important;
    border-radius: 14px !important;
    color: #fff !important;
    background: linear-gradient(180deg, #24cfa2, #0d9b78 60%, #08725b) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,.36) inset, 0 -4px 0 rgba(3,68,51,.42) inset, 0 10px 24px rgba(8,139,105,.27) !important;
    font-weight: 800 !important;
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
}

.roam-key-submit:hover {
    filter: brightness(1.06);
    box-shadow: 0 1px 0 rgba(255,255,255,.4) inset, 0 -4px 0 rgba(3,68,51,.42) inset, 0 16px 30px rgba(8,139,105,.34) !important;
    transform: translateY(-3px) rotateX(4deg);
}

.roam-key-submit:active { transform: translateY(2px) !important; }

.roam-key-footnote {
    position: relative;
    margin: 15px 0 0;
    padding-top: 13px;
    border-top: 1px solid rgba(17,103,81,.1);
    color: #72878a;
    font-size: 10.5px;
    line-height: 1.6;
}

@keyframes roam-key-fade {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes roam-key-arrive {
    from { opacity: 0; transform: translateY(20px) rotateX(8deg) scale(.96); }
    to { opacity: 1; transform: translateY(0) rotateX(0) scale(1); }
}

.gradio-container {
    width: 100% !important; max-width: none !important; min-width: 0 !important;
    padding: 32px 28px 56px !important; box-sizing: border-box !important;
}

.gradio-container .main, .gradio-container .wrap {
    width: 100% !important; min-width: 0 !important; max-width: 100%;
}

.gradio-container .main { margin: 0 auto !important; }
.gradio-container main.contain { max-width: 1480px !important; margin: 0 auto !important; }

/* ════════════════════════════════
   HERO — Full 3D Card
   ════════════════════════════════ */
.roam-hero {
    position: relative;
    display: flex;
    min-height: 350px;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    overflow: hidden;
    padding: 50px 60px;
    border: 1px solid rgba(255,255,255,.26);
    border-radius: 32px;
    color: #fff;
    background:
        radial-gradient(ellipse at 78% 42%, rgba(14,168,154,.28), transparent 30rem),
        radial-gradient(ellipse at 20% 80%, rgba(11,100,90,.22), transparent 20rem),
        linear-gradient(118deg, #0a1e28 0%, #113d3a 42%, #0d5a50 72%, #1a7a68 100%);
    box-shadow:
        0 2px 0 rgba(255,255,255,.12) inset,
        0 -2px 0 rgba(0,0,0,.18) inset,
        0 40px 80px rgba(10,40,38,.35),
        0 8px 24px rgba(14,168,154,.12);
    isolation: isolate;
    perspective: var(--roam-perspective);
    transform-style: preserve-3d;
    animation: roam-rise .9s cubic-bezier(.15,.7,.2,1) both;
    transition: transform .4s ease, box-shadow .4s ease;
}

.roam-hero:hover {
    transform: translateY(-4px) rotateX(1deg);
    box-shadow:
        0 2px 0 rgba(255,255,255,.14) inset,
        0 -2px 0 rgba(0,0,0,.2) inset,
        0 54px 96px rgba(10,40,38,.4),
        0 12px 36px rgba(14,168,154,.18);
}

/* dot-grid overlay */
.roam-hero:before {
    position: absolute; z-index: -1; inset: 0;
    background-image: radial-gradient(rgba(255,255,255,.5) .9px, transparent .9px);
    background-size: 28px 28px;
    content: "";
    mask-image: linear-gradient(100deg, transparent 15%, #000 90%);
    opacity: .12;
}

/* large decorative ring */
.roam-hero:after {
    content: "";
    position: absolute; z-index: -1;
    width: 480px; height: 480px;
    right: -210px; top: -270px;
    border: 1px solid rgba(14,240,200,.1);
    border-radius: 50%;
    box-shadow:
        0 0 0 44px rgba(14,240,200,.03),
        0 0 0 100px rgba(14,240,200,.018),
        0 0 80px rgba(14,168,154,.12);
}

/* neon bottom-edge glow line */
.roam-hero-glow {
    position: absolute;
    bottom: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(14,240,200,.55), transparent);
    filter: blur(1px);
}

.roam-hero-copy { position: relative; z-index: 2; max-width: 660px; }

.roam-hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 24px;
}

.roam-hero-meta span {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 11px;
    border: 1px solid rgba(200,255,235,.18);
    border-radius: 999px;
    color: rgba(235,255,247,.88);
    background: linear-gradient(145deg, rgba(255,255,255,.13), rgba(255,255,255,.045));
    box-shadow: 0 7px 18px rgba(0,0,0,.12), inset 0 1px rgba(255,255,255,.14);
    backdrop-filter: blur(12px);
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.roam-hero-meta span:before {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #75ffd0;
    box-shadow: 0 0 10px rgba(117,255,208,.85);
    content: "";
}

.roam-hero-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
    margin-top: 25px;
}

.roam-hero-cta {
    display: inline-flex;
    min-height: 48px;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 0 19px;
    border: 1px solid rgba(194,255,231,.42);
    border-radius: 15px;
    color: #07372f !important;
    background: linear-gradient(180deg, #c8ffe8 0%, #76e9c2 48%, #42cba2 100%);
    box-shadow:
        0 1px 0 rgba(255,255,255,.75) inset,
        0 -4px 0 rgba(8,94,71,.28) inset,
        0 12px 26px rgba(0,0,0,.2),
        0 0 24px rgba(88,255,202,.13);
    font-size: 12px;
    font-weight: 850;
    letter-spacing: .04em;
    text-decoration: none !important;
    text-shadow: 0 1px rgba(255,255,255,.35);
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}

.roam-hero-cta span {
    display: grid;
    width: 23px;
    height: 23px;
    place-items: center;
    border-radius: 8px;
    color: #dffff2;
    background: linear-gradient(145deg, #126b59, #084538);
    box-shadow: 0 3px 8px rgba(4,54,42,.28), inset 0 1px rgba(255,255,255,.22);
    font-size: 15px;
    transition: transform .18s ease;
}

.roam-hero-cta:hover {
    filter: brightness(1.06);
    box-shadow:
        0 1px 0 rgba(255,255,255,.82) inset,
        0 -4px 0 rgba(8,94,71,.28) inset,
        0 17px 32px rgba(0,0,0,.24),
        0 0 30px rgba(88,255,202,.22);
    transform: translateY(-4px) rotateX(5deg);
}

.roam-hero-cta:hover span { transform: translateY(2px); }

.roam-hero-cta:active {
    box-shadow: 0 1px 0 rgba(255,255,255,.55) inset, 0 3px 8px rgba(0,0,0,.18);
    transform: translateY(2px);
}

.roam-hero-cta:focus-visible {
    outline: 3px solid rgba(200,255,235,.85);
    outline-offset: 4px;
}

/* ════════════════════════════════
   GLOBE ART — Full 3D
   ════════════════════════════════ */
.roam-hero-art {
    position: relative;
    display: grid;
    width: 340px; height: 290px;
    flex: 0 0 340px;
    place-items: center;
    perspective: 1000px;
    transform-style: preserve-3d;
    animation: roam-float 7s ease-in-out infinite;
}

/* Floating 3D destination widgets give the hero a layered travel-dashboard feel. */
.roam-float-card {
    position: absolute;
    z-index: 4;
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 146px;
    padding: 11px 13px;
    border: 1px solid rgba(207,255,237,.42);
    border-radius: 15px;
    color: #effff8;
    background: linear-gradient(145deg, rgba(255,255,255,.2), rgba(12,54,54,.76));
    box-shadow: 0 1px 0 rgba(255,255,255,.22) inset, 0 14px 28px rgba(0,0,0,.23), 0 0 22px rgba(52,226,186,.12);
    backdrop-filter: blur(12px);
    transform-style: preserve-3d;
    animation: roam-card-drift 6s ease-in-out infinite;
    pointer-events: none;
}
.roam-float-card strong { display: block; color: #fff; font-size: 11px; letter-spacing: .02em; }
.roam-float-card small { display: block; margin-top: 2px; color: rgba(226,255,242,.72); font-size: 9px; }
.roam-float-icon {
    display: grid;
    width: 31px; height: 31px;
    flex: 0 0 31px;
    place-items: center;
    border: 1px solid rgba(212,255,240,.28);
    border-radius: 10px;
    background: linear-gradient(145deg, #31d7ac, #0a786e);
    box-shadow: 0 3px 0 rgba(2,52,47,.35), 0 7px 14px rgba(0,0,0,.18);
    font-size: 15px;
    transform: translateZ(16px) rotate(-7deg);
}
.roam-float-card-one { top: 18px; left: -4px; transform: rotateY(12deg) rotateZ(-5deg); }
.roam-float-card-two { right: -6px; bottom: 20px; animation-delay: -2.4s; transform: rotateY(-12deg) rotateZ(4deg); }
@keyframes roam-card-drift {
    0%,100% { margin-top: 0; filter: brightness(1); }
    50% { margin-top: -10px; filter: brightness(1.12); }
}


/* A shaded sphere with a moving, repeating surface map. */
.roam-globe {
    position: relative;
    width: 200px; height: 200px;
    overflow: hidden;
    border: 1px solid rgba(180,255,230,.35);
    border-radius: 50%;
    background:
        radial-gradient(ellipse 42% 42% at 30% 26%, rgba(76,226,187,.46), transparent 76%),
        radial-gradient(circle at 34% 30%, #147f77 0%, #0b5557 48%, #062735 78%, #03131e 100%);
    box-shadow:
        inset -30px -20px 46px rgba(0,8,20,.78),
        inset 12px 10px 24px rgba(193,255,231,.2),
        0 0 0 7px rgba(61,227,186,.035),
        0 0 36px rgba(14,200,165,.26),
        0 22px 48px rgba(2,14,26,.48);
    transform-style: preserve-3d;
    animation: roam-globe-rock 9s ease-in-out infinite alternate;
    will-change: transform;
}

.roam-globe-map {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 200%;
    display: flex;
    transform: translateZ(1px);
    animation: roam-globe-surface 24s linear infinite;
    will-change: transform;
}

.roam-globe-map svg {
    display: block;
    width: 50%;
    height: 100%;
    flex: 0 0 50%;
}

.roam-globe-map .roam-land {
    fill: #75c98a;
    stroke: rgba(195,255,205,.48);
    stroke-width: 1;
}

.roam-globe-map .roam-land-highlight { fill: #a2df98; }

.roam-globe-map .roam-grid {
    fill: none;
    stroke: rgba(172,255,233,.26);
    stroke-width: .8;
}

.roam-globe-shade {
    position: absolute;
    z-index: 1;
    inset: 0;
    border-radius: inherit;
    transform: translateZ(3px);
    background:
        radial-gradient(ellipse 26% 20% at 29% 21%, rgba(255,255,255,.63), rgba(191,255,231,.18) 42%, transparent 100%),
        radial-gradient(ellipse at 50% 50%, transparent 48%, rgba(0,13,24,.24) 72%, rgba(0,7,17,.74) 100%),
        linear-gradient(112deg, rgba(174,255,231,.12), transparent 39%, rgba(0,8,20,.34) 86%);
    box-shadow:
        inset 0 0 0 1px rgba(207,255,238,.18),
        inset -27px -14px 36px rgba(0,6,17,.36);
    pointer-events: none;
}

/* 3D atmosphere glow ring */
.roam-globe-atm {
    position: absolute;
    width: 230px; height: 230px;
    border-radius: 50%;
    border: 1px solid rgba(14,240,200,.18);
    box-shadow:
        0 0 24px rgba(14,240,200,.14),
        inset 0 0 24px rgba(14,240,200,.08);
    pointer-events: none;
}

/* Orbits */
.roam-orbit {
    position: absolute;
    width: 300px; height: 118px;
    border: 1.5px solid rgba(195,255,230,.52);
    border-radius: 50%;
    box-shadow: 0 0 18px rgba(14,240,200,.14);
    transform: rotate(-23deg) rotateX(66deg);
    animation: roam-orbit-spin 16s linear infinite;
    transform-style: preserve-3d;
}

/* satellite dot */
.roam-orbit:after {
    position: absolute;
    top: calc(50% - 5px); right: 10%;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #fff, #8df0c3);
    box-shadow: 0 0 16px 5px rgba(174,255,223,.78);
    content: "";
    animation: roam-orbit-glow 2.4s ease-in-out infinite;
}

.roam-orbit-two {
    width: 258px; height: 160px;
    border-color: rgba(195,255,230,.2);
    transform: rotate(34deg) rotateX(67deg);
    animation-name: roam-orbit-spin-two;
    animation-duration: 22s;
}

/* location pins */
.roam-pin {
    position: absolute; z-index: 2;
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px;
    border: 1px solid rgba(255,255,255,.44);
    border-radius: 14px;
    color: #e8fff6;
    background: rgba(10,38,46,.58);
    box-shadow:
        0 12px 28px rgba(2,20,28,.28),
        0 2px 8px rgba(14,168,154,.14),
        inset 0 1px rgba(255,255,255,.22);
    backdrop-filter: blur(14px);
    font-size: 11px; font-weight: 700; letter-spacing: .03em;
    animation: roam-pin-pop .7s cubic-bezier(.2,.9,.3,1) both;
    transition: transform .25s ease, box-shadow .25s ease;
}

.roam-pin:hover {
    transform: scale(1.06) translateY(-2px);
    box-shadow:
        0 16px 32px rgba(2,20,28,.32),
        0 0 20px rgba(14,240,200,.18),
        inset 0 1px rgba(255,255,255,.26);
}

/* glowing dot */
.roam-pin:before {
    flex-shrink: 0;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6dffc4;
    box-shadow: 0 0 8px 2px rgba(109,255,196,.8);
    content: "";
    animation: roam-pin-blink 2s ease-in-out infinite;
}

.roam-pin-one { top: 22px; right: -4px; animation-delay: .4s; }
.roam-pin-two { bottom: 22px; left: -4px; animation-delay: .65s; }

/* ════════════════════════════════
   TYPOGRAPHY
   ════════════════════════════════ */
.roam-eyebrow {
    margin-bottom: 16px;
    color: #9adfc6;
    font-size: 11px; font-weight: 800; letter-spacing: .2em;
    text-transform: uppercase;
}

.roam-hero h1 {
    margin: 0;
    color: white;
    font-size: clamp(38px, 5.4vw, 64px);
    line-height: 1.04;
    letter-spacing: -.046em;
    text-shadow:
        0 2px 0 rgba(14,168,154,.3),
        0 6px 28px rgba(4,22,30,.28);
}

.roam-hero p {
    max-width: 540px;
    margin: 20px 0 0;
    color: rgba(255,255,255,.8);
    font-size: 15px; line-height: 1.78;
}

.roam-section-title {
    margin: 32px 0 16px;
    color: var(--roam-ink);
    font-size: 12px; font-weight: 800; letter-spacing: .14em;
    text-transform: uppercase;
}

/* ════════════════════════════════
   SERVICE CARDS — 3D tilt
   ════════════════════════════════ */
.roam-services {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
    perspective: var(--roam-perspective);
}

.roam-service {
    position: relative;
    min-height: 138px;
    padding: 22px 24px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.9);
    border-radius: 22px;
    background: linear-gradient(148deg, rgba(255,255,255,.95), rgba(240,251,246,.75));
    box-shadow:
        0 1px 0 rgba(255,255,255,.95) inset,
        0 -1px 0 rgba(14,168,154,.06) inset,
        0 14px 36px rgba(14,80,70,.08),
        0 2px 8px rgba(14,168,154,.05);
    transform: translateZ(0) perspective(600px);
    transform-style: preserve-3d;
    transition: transform .38s cubic-bezier(.15,.8,.2,1), box-shadow .38s ease, border-color .38s ease;
    animation: roam-rise .75s cubic-bezier(.2,.7,.2,1) both;
    cursor: default;
}

.roam-service:nth-child(2) { animation-delay: .12s; }
.roam-service:nth-child(3) { animation-delay: .24s; }

/* decorative orb top-right */
.roam-service:after {
    position: absolute;
    top: -52px; right: -48px;
    width: 128px; height: 128px;
    border: 1px solid rgba(14,168,154,.1);
    border-radius: 50%;
    box-shadow:
        0 0 0 16px rgba(14,168,154,.022),
        0 0 0 34px rgba(14,168,154,.015);
    content: "";
    transition: transform .48s ease, opacity .3s ease;
    opacity: .7;
}

/* glass top-sheen */
.roam-service:before {
    position: absolute;
    inset: 0 0 70% 0;
    border-radius: 22px 22px 0 0;
    background: linear-gradient(180deg, rgba(255,255,255,.55), transparent);
    content: "";
    pointer-events: none;
}

.roam-service:hover {
    border-color: rgba(14,168,154,.28);
    box-shadow:
        0 1px 0 rgba(255,255,255,.95) inset,
        0 -1px 0 rgba(14,168,154,.1) inset,
        0 28px 52px rgba(14,80,70,.16),
        0 0 0 1px rgba(14,168,154,.08),
        0 4px 16px rgba(14,168,154,.1);
    transform: translateY(-10px) rotateX(4deg) rotateY(-3deg) scale(1.012);
}

.roam-service:hover:after {
    transform: scale(1.22) translate(-6px, 6px);
    opacity: 1;
}

/* number badge — 3D extruded */
.roam-service-icon {
    display: inline-flex;
    width: 34px; height: 34px;
    align-items: center; justify-content: center;
    margin-bottom: 14px;
    border: 1px solid rgba(14,168,154,.12);
    border-radius: 12px;
    color: #0b6e66;
    background: linear-gradient(148deg, #eefbf5, #cff0e4);
    box-shadow:
        0 4px 12px rgba(14,168,154,.1),
        0 1px 0 rgba(255,255,255,.9) inset,
        0 -1px 0 rgba(14,168,154,.08) inset;
    font-size: 12px; font-weight: 900;
    transform: translateZ(8px);
    transition: transform .3s ease, box-shadow .3s ease;
}

.roam-service:hover .roam-service-icon {
    transform: translateZ(16px) scale(1.08);
    box-shadow:
        0 8px 20px rgba(14,168,154,.18),
        0 1px 0 rgba(255,255,255,.9) inset;
}

.roam-service strong {
    display: block;
    color: var(--roam-ink);
    font-size: 14px; font-weight: 750;
    transform: translateZ(4px);
}

.roam-service span:last-child {
    display: block;
    margin-top: 5px;
    color: var(--roam-muted);
    font-size: 12.5px; line-height: 1.5;
}

/* ════════════════════════════════
   CHAT PANEL — 3D frosted card
   ════════════════════════════════ */
.roam-chat {
    width: 100% !important; min-width: 0 !important;
    margin-top: 26px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.92) !important;
    border-radius: 28px !important;
    background: rgba(255,255,255,.82) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.98) inset,
        0 -1px 0 rgba(14,168,154,.07) inset,
        0 28px 64px rgba(14,70,62,.13),
        0 4px 16px rgba(14,168,154,.07);
    backdrop-filter: blur(20px) saturate(1.5);
    transition: box-shadow .32s ease, transform .32s ease;
}

.roam-chat:focus-within {
    box-shadow:
        0 1px 0 rgba(255,255,255,.98) inset,
        0 -1px 0 rgba(14,168,154,.12) inset,
        0 34px 76px rgba(14,70,62,.17),
        0 0 0 2px rgba(14,168,154,.16),
        0 4px 16px rgba(14,168,154,.1);
    transform: translateY(-3px);
}

.roam-chat .bubble-wrap {
    background:
        radial-gradient(ellipse at 50% 15%, rgba(200,245,228,.5), transparent 55%),
        linear-gradient(180deg, rgba(248,253,250,.99), rgba(238,250,244,.96)) !important;
}

/* ── Example prompt cards ── */
.roam-chat .examples { gap: 14px !important; padding: 20px 20px 16px !important; }

.roam-examples {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr);
    gap: 12px !important;
    margin: 4px 16px 18px !important;
}

.roam-examples button {
    min-height: 66px !important;
    padding: 13px 15px !important;
    border: 1px solid rgba(14,168,154,.16) !important;
    border-radius: 16px !important;
    color: #163840 !important;
    background: linear-gradient(150deg, #fff, #eaf8f3) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,.98) inset, 0 -2px 0 rgba(14,168,154,.06) inset, 0 8px 20px rgba(14,80,70,.07) !important;
    font-size: 12px !important;
    font-weight: 650 !important;
    line-height: 1.5 !important;
    text-align: left !important;
    white-space: normal !important;
    transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease !important;
}

.roam-examples button:hover {
    border-color: rgba(14,168,154,.4) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,.98) inset, 0 -3px 0 rgba(14,168,154,.1) inset, 0 16px 30px rgba(14,80,70,.14) !important;
    transform: translateY(-4px) rotateX(3deg) !important;
}

.roam-chat .example {
    min-height: 78px !important;
    padding: 15px 18px !important;
    border: 1px solid rgba(14,168,154,.16) !important;
    border-radius: 18px !important;
    color: #163840 !important;
    background: linear-gradient(150deg, #ffffff 0%, #eaf8f3 100%) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.98) inset,
        0 -2px 0 rgba(14,168,154,.06) inset,
        0 8px 20px rgba(14,80,70,.07) !important;
    font-size: 13px !important; font-weight: 650 !important; line-height: 1.52 !important;
    text-align: left !important;
    transition: transform .28s cubic-bezier(.15,.8,.2,1), box-shadow .28s ease, border-color .28s ease !important;
}

.roam-chat .example:hover {
    border-color: rgba(14,168,154,.4) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.98) inset,
        0 -3px 0 rgba(14,168,154,.1) inset,
        0 20px 36px rgba(14,80,70,.15),
        0 0 0 1px rgba(14,168,154,.1) !important;
    transform: translateY(-7px) rotateX(4deg) rotateY(-1.5deg) scale(1.025) !important;
}

/* focus rings */
.roam-chat .example:focus-visible,
.roam-chat .submit-button:focus-visible,
.roam-chat .stop-button:focus-visible {
    outline: 2px solid rgba(14,168,154,.55) !important;
    outline-offset: 3px !important;
}

/* ════════════════════════════════
   3D SUBMIT BUTTON — extruded slab
   ════════════════════════════════ */
.roam-chat .submit-button,
.roam-chat .stop-button {
    position: relative !important;
    min-width: 148px !important; min-height: 50px !important;
    padding: 13px 24px !important;
    border: none !important;
    border-radius: 15px !important;
    color: #fff !important;
    /* multi-stop gradient — top lighter, bottom deeper */
    background: linear-gradient(180deg,
        #22c9a2 0%,
        #0fa88e 35%,
        #0a8a74 70%,
        #07705e 100%) !important;
    box-shadow:
        /* top sheen */
        0 1px 0 rgba(255,255,255,.32) inset,
        /* bottom pressed edge — gives 3D depth */
        0 -5px 0 rgba(4,52,44,.55) inset,
        /* left side light */
        1px 0 0 rgba(255,255,255,.12) inset,
        /* outer glow */
        0 10px 28px rgba(14,168,154,.42),
        0 2px 8px rgba(14,168,154,.25) !important;
    font-size: 13.5px !important; font-weight: 800 !important; letter-spacing: .025em !important;
    text-shadow: 0 1px 3px rgba(2,38,32,.4) !important;
    transition: transform .16s cubic-bezier(.2,.8,.2,1), box-shadow .16s ease, filter .16s ease !important;
    will-change: transform;
}

.roam-chat .submit-button:hover,
.roam-chat .stop-button:hover {
    filter: brightness(1.08) saturate(1.1) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.36) inset,
        0 -5px 0 rgba(4,52,44,.55) inset,
        1px 0 0 rgba(255,255,255,.14) inset,
        0 16px 36px rgba(14,168,154,.52),
        0 4px 14px rgba(14,168,154,.3) !important;
    transform: translateY(-4px) !important;
}

.roam-chat .submit-button:active,
.roam-chat .stop-button:active {
    filter: brightness(.97) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.22) inset,
        0 -2px 0 rgba(4,52,44,.55) inset,
        0 4px 10px rgba(14,168,154,.22) !important;
    transform: translateY(2px) !important;
}

/* ════════════════════════════════
   3D SEARCH / TEXTAREA BAR
   ════════════════════════════════ */
.roam-chat textarea,
.roam-chat input[type="text"] {
    min-height: 52px !important;
    padding: 15px 18px !important;
    border: 1.5px solid rgba(14,168,154,.18) !important;
    border-radius: 16px !important;
    /* inset gradient — looks like it's pressed into the page */
    background: linear-gradient(180deg, #eaf6f2 0%, #f4fbf8 60%, #ffffff 100%) !important;
    box-shadow:
        /* top inset shadow — depth illusion */
        0 2px 6px rgba(14,80,70,.08) inset,
        0 1px 0 rgba(255,255,255,.95),
        /* outer subtle shadow */
        0 2px 10px rgba(14,80,70,.04) !important;
    font-size: 14px !important;
    color: #0f1f28 !important;
    transition: border-color .22s ease, box-shadow .22s ease, transform .22s ease, background .22s ease !important;
}

.roam-chat textarea:focus,
.roam-chat input[type="text"]:focus {
    border-color: rgba(14,168,154,.56) !important;
    background: linear-gradient(180deg, #e4f4ef 0%, #f0faf6 60%, #fafffe 100%) !important;
    box-shadow:
        /* deeper inset on focus */
        0 3px 8px rgba(14,80,70,.12) inset,
        0 1px 0 rgba(255,255,255,.95),
        /* focus ring glow */
        0 0 0 3px rgba(14,168,154,.14),
        0 6px 20px rgba(14,80,70,.08) !important;
    transform: translateY(-1px) !important;
    outline: none !important;
}

/* text input container row */
.roam-chat .input-row,
.roam-chat form > div:last-child {
    background: linear-gradient(0deg, rgba(224,246,238,.92), rgba(245,252,249,.94)) !important;
    padding: 14px 16px !important;
    border-top: 1px solid rgba(14,168,154,.09) !important;
    box-shadow: 0 -1px 0 rgba(255,255,255,.8) !important;
}

/* ════════════════════════════════
   FOOTER
   ════════════════════════════════ */
.roam-note {
    margin: 18px 4px 0;
    color: var(--roam-muted);
    font-size: 11.5px; line-height: 1.65;
}

/* Keep custom app surfaces in sync with Gradio's Settings theme switch. */
body.dark {
    --roam-ink: #e7f2f1;
    --roam-muted: #a4bcba;
    --roam-card-bg: rgba(15, 36, 39, .94);
    color-scheme: dark;
}

body.dark,
body.dark gradio-app,
body.dark .gradio-container {
    color: #e7f2f1 !important;
    background:
        radial-gradient(ellipse at 10% 0%, rgba(14,168,154,.12), transparent 36rem),
        radial-gradient(ellipse at 92% 55%, rgba(11,122,110,.12), transparent 34rem),
        linear-gradient(165deg, #071619 0%, #0b1d20 48%, #101f21 100%) !important;
}

body.dark .gradio-container.gradio-container .contain .roam-service {
    border-color: rgba(145, 202, 190, .18) !important;
    background: linear-gradient(148deg, rgba(23,48,49,.97), rgba(15,37,39,.94)) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.04) inset,
        0 -1px 0 rgba(14,168,154,.08) inset,
        0 14px 36px rgba(0,0,0,.2);
}

body.dark .gradio-container.gradio-container .contain .roam-service-icon {
    border-color: rgba(66, 196, 165, .22) !important;
    color: #a9f3d9 !important;
    background: linear-gradient(148deg, #1d4947, #153a39) !important;
    box-shadow:
        0 4px 12px rgba(0,0,0,.18),
        0 1px 0 rgba(255,255,255,.06) inset;
}

body.dark .gradio-container.gradio-container .contain .roam-chat {
    border-color: rgba(145, 202, 190, .18) !important;
    background: rgba(13, 30, 33, .92) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.04) inset,
        0 28px 64px rgba(0,0,0,.28);
}

body.dark .gradio-container.gradio-container .contain .roam-chat .bubble-wrap {
    color: #e7f2f1 !important;
    background:
        radial-gradient(ellipse at 50% 15%, rgba(28,85,72,.25), transparent 55%),
        linear-gradient(180deg, rgba(17,39,40,.99), rgba(13,31,33,.98)) !important;
}

body.dark .gradio-container.gradio-container .contain .roam-examples button,
body.dark .gradio-container.gradio-container .contain .roam-chat .example {
    border-color: rgba(111, 190, 171, .22) !important;
    color: #e1efed !important;
    background: linear-gradient(150deg, #1a3839, #122b2d) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.05) inset,
        0 -2px 0 rgba(14,168,154,.08) inset,
        0 8px 20px rgba(0,0,0,.2) !important;
}

body.dark .gradio-container.gradio-container .contain .roam-examples button:hover,
body.dark .gradio-container.gradio-container .contain .roam-chat .example:hover {
    border-color: rgba(79, 212, 178, .52) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,.06) inset,
        0 -3px 0 rgba(14,168,154,.13) inset,
        0 16px 30px rgba(0,0,0,.3) !important;
}

body.dark .gradio-container.gradio-container .contain .roam-chat textarea,
body.dark .gradio-container.gradio-container .contain .roam-chat input[type="text"] {
    border-color: rgba(111, 190, 171, .24) !important;
    color: #e7f2f1 !important;
    background: linear-gradient(180deg, #142d2f 0%, #102628 60%, #0d2224 100%) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,.22) inset, 0 1px 0 rgba(255,255,255,.04) !important;
}

body.dark .gradio-container.gradio-container .contain .roam-chat textarea:focus,
body.dark .gradio-container.gradio-container .contain .roam-chat input[type="text"]:focus {
    background: linear-gradient(180deg, #193638 0%, #142f31 60%, #112a2c 100%) !important;
    box-shadow:
        0 3px 8px rgba(0,0,0,.24) inset,
        0 0 0 3px rgba(14,168,154,.2),
        0 6px 20px rgba(0,0,0,.18) !important;
}

body.dark .gradio-container.gradio-container .contain .roam-chat .input-row,
body.dark .gradio-container.gradio-container .contain .roam-chat form > div:last-child {
    border-color: rgba(111, 190, 171, .16) !important;
    background: linear-gradient(0deg, rgba(12,31,33,.98), rgba(18,39,40,.98)) !important;
    box-shadow: 0 -1px 0 rgba(255,255,255,.04) !important;
}

/* ════════════════════════════════
   KEYFRAMES
   ════════════════════════════════ */
@keyframes roam-rise {
    from { opacity: 0; transform: translateY(22px) rotateX(4deg); }
    to   { opacity: 1; transform: translateY(0)   rotateX(0deg); }
}

@keyframes roam-float {
    0%,100% { transform: translateY(0)     rotateZ(-2.5deg) rotateX(2deg); }
    50%      { transform: translateY(-13px) rotateZ(2.5deg)  rotateX(-2deg); }
}

@keyframes roam-globe-surface {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}

@keyframes roam-globe-rock {
    0% { transform: rotateX(7deg) rotateY(-12deg) rotateZ(-3deg) scale(.98); }
    100% { transform: rotateX(-5deg) rotateY(12deg) rotateZ(3deg) scale(1.02); }
}

@keyframes roam-orbit-spin {
    from { transform: rotate(-23deg) rotateX(66deg); }
    to   { transform: rotate(337deg) rotateX(66deg); }
}

@keyframes roam-orbit-spin-two {
    from { transform: rotate(34deg)  rotateX(67deg); }
    to   { transform: rotate(-326deg) rotateX(67deg); }
}

@keyframes roam-orbit-glow {
    0%,100% { opacity: .6;  transform: scale(1);    }
    50%      { opacity: 1;   transform: scale(1.3);  }
}

@keyframes roam-pin-pop {
    from { opacity: 0; transform: scale(.6) translateY(8px); }
    to   { opacity: 1; transform: scale(1)  translateY(0);   }
}

@keyframes roam-pin-blink {
    0%,100% { opacity: 1;   box-shadow: 0 0 8px 2px rgba(109,255,196,.8); }
    50%      { opacity: .55; box-shadow: 0 0 4px 1px rgba(109,255,196,.4); }
}

/* particle trail on globe */
@keyframes roam-particle {
    0%   { opacity: 0; transform: translate(0,0) scale(.6); }
    30%  { opacity: 1; }
    100% { opacity: 0; transform: translate(var(--tx,20px), var(--ty,-20px)) scale(.2); }
}

/* ════════════════════════════════
   RESPONSIVE
   ════════════════════════════════ */
@media (max-width: 700px) {
    .gradio-container { padding: 14px 12px 32px !important; }
    .roam-key-card { padding: 24px !important; border-radius: 23px !important; }
    .roam-hero { min-height: 0; flex-direction: column; align-items: stretch; gap: 0; padding: 32px 26px 24px; border-radius: 24px; }
    .roam-hero-copy { width: 100%; }
    .roam-hero-actions { margin-top: 20px; }
    .roam-hero-art { width: 100%; height: 200px; flex-basis: 200px; margin-top: 4px; }
    .roam-globe { width: 148px; height: 148px; }
    .roam-globe-atm { width: 174px; height: 174px; }
    .roam-orbit { width: 228px; height: 90px; }
    .roam-orbit-two { width: 200px; height: 126px; }
    .roam-pin-one { top: 10px; right: 0; }
    .roam-pin-two { bottom: 8px; left: 2px; }
    .roam-float-card { min-width: 132px; padding: 9px 10px; }
    .roam-float-card-one { top: 2px; left: -2px; }
    .roam-float-card-two { right: -2px; bottom: 0; }
    .roam-services { grid-template-columns: 1fr; gap: 11px; }
    .roam-service { min-height: 0; padding: 18px 20px; }
    .roam-examples { grid-template-columns: 1fr !important; margin: 4px 10px 14px !important; }
    .roam-chat .examples { grid-template-columns: 1fr !important; padding: 13px !important; }
    .roam-chat .example { min-height: 0 !important; }
    .roam-chat .submit-button, .roam-chat .stop-button { min-width: 118px !important; }
}

@media (min-width: 701px) and (max-width: 920px) {
    .roam-hero { padding: 38px 36px; }
    .roam-hero-art { width: 270px; flex-basis: 270px; }
    .roam-globe { width: 158px; height: 158px; }
    .roam-globe-atm { width: 185px; height: 185px; }
}

@media (prefers-reduced-motion: reduce) {
    *, *:before, *:after {
        scroll-behavior: auto !important;
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="md",
)

# Hugging Face currently provisions this Space on ZeroGPU. The app itself
# does not need GPU compute, but ZeroGPU requires at least one registered
# @spaces.GPU Gradio event handler before startup.
@spaces.GPU(duration=1)
def _zerogpu_startup_compatibility() -> str:
    """No-op handler registered only to satisfy the ZeroGPU startup check."""
    return "Roamly does not require GPU compute."


with gr.Blocks(
    title="Roamly | AI Travel Studio",
    fill_width=True,
) as demo:
    # Keep the compatibility event hidden; normal visitors never need to run it.
    zerogpu_compatibility_button = gr.Button(visible=False)
    zerogpu_compatibility_button.click(
        fn=_zerogpu_startup_compatibility,
        inputs=None,
        outputs=None,
        api_name=False,
    )
    openai_api_key_state = gr.State(value="")
    tavily_api_key_state = gr.State(value="")
    with gr.Column(elem_id="api-key-gate", elem_classes=["roam-key-gate"]) as api_key_gate:
        with gr.Column(elem_classes=["roam-key-card"]):
            gr.HTML(
                """
                <div class="roam-key-icon" aria-hidden="true"><span></span></div>
                <h2>Unlock your trip planner</h2>
                <p class="roam-key-copy">
                    Add your OpenAI key to start planning. A Tavily key is optional
                    and enables live web search. Keys are requested again in a new session.
                </p>
                """
            )
            openai_api_key_input = gr.Textbox(
                label="OpenAI API key",
                placeholder="sk-...",
                type="password",
                show_label=True,
                html_attributes=gr.InputHTMLAttributes(
                    autocomplete="new-password",
                    autocapitalize="off",
                    spellcheck=False,
                ),
            )
            tavily_api_key_input = gr.Textbox(
                label="Tavily API key (optional — enables live web search)",
                placeholder="tvly-...",
                type="password",
                show_label=True,
                html_attributes=gr.InputHTMLAttributes(
                    autocomplete="new-password",
                    autocapitalize="off",
                    spellcheck=False,
                ),
            )
            trust_confirmation = gr.Checkbox(
                label="I trust this app with my OpenAI and optional Tavily API keys",
                value=False,
                elem_classes=["roam-key-trust"],
            )
            verify_key_button = gr.Button(
                "Verify key & continue",
                variant="primary",
                elem_classes=["roam-key-submit"],
            )
            gr.HTML(
                """
                <p class="roam-key-footnote">
                    Your keys are sent to this app's server for OpenAI and (if supplied)
                    Tavily requests. They stay in session memory and are not saved to
                    browser storage or files. Only enter keys on a deployment you trust.
                    You can leave Tavily blank and still use mapped-place discovery.
                </p>
                """
            )
    verify_key_button.click(
        fn=verify_openai_api_key,
        inputs=[openai_api_key_input, tavily_api_key_input, trust_confirmation],
        outputs=[
            openai_api_key_state,
            tavily_api_key_state,
            openai_api_key_input,
            tavily_api_key_input,
            api_key_gate,
        ],
        api_name=False,
    )

    gr.HTML(
        """
        <header class="roam-hero">
            <div class="roam-hero-copy">
                <div class="roam-eyebrow">✦ Your next trip starts here</div>
                <h1>Travel more thoughtfully.</h1>
                <p>
                    Plan a getaway with a personal AI travel assistant. Discover
                    local places, check current travel information, and build a
                    practical itinerary in one conversation.
                </p>
                <div class="roam-hero-meta" aria-label="Travel planning features">
                    <span>AI trip planner</span>
                    <span>Optional live web search</span>
                    <span>Open map data</span>
                </div>
                <div class="roam-hero-actions">
                    <a class="roam-hero-cta" href="#trip-planner">
                        Start planning <span aria-hidden="true">↓</span>
                    </a>
                </div>
            </div>
            <div class="roam-hero-art" aria-hidden="true">
                <div class="roam-orbit"></div>
                <div class="roam-orbit roam-orbit-two"></div>
                <div class="roam-globe-atm"></div>
                <div class="roam-globe">
                    <div class="roam-globe-map">
                        <svg viewBox="0 0 200 200" aria-hidden="true">
                            <g class="roam-land">
                                <path class="roam-land-highlight" d="M19 41 29 29 43 26 49 19 63 22 70 31 66 39 74 45 67 53 60 55 56 66 49 69 44 61 36 58 31 50 24 48Z"/>
                                <path d="m52 71 8 5 5 12-4 12-5 12-3 15-6-4-2-12-5-10 2-12-5-8 7-6Z"/>
                                <path d="m82 36 7-7 12 2 4 7-8 5-9-1Z"/>
                                <path class="roam-land-highlight" d="m91 54 9-5 10 3 8 10-2 12-6 7-2 13-7 13-6 16-7-2-3-13-7-10 1-13-5-11 5-12Z"/>
                                <path d="m105 38 11-9 19-4 13 5 14-1 16 8-3 8-13 2-8 7-12-3-8 8-10-4-6-9-11 1-5-5Z"/>
                                <path class="roam-land-highlight" d="m145 81 10 2 7 8-4 8-8-2-6-7Z"/>
                                <path d="m160 117 12-2 10 8-5 10-13 2-7-8Z"/>
                            </g>
                            <g class="roam-grid">
                                <path d="M0 52 Q100 67 200 52M0 100 Q100 112 200 100M0 148 Q100 133 200 148"/>
                                <path d="M45 0 C76 52 76 148 45 200M100 0 C116 52 116 148 100 200M155 0 C124 52 124 148 155 200"/>
                            </g>
                        </svg>
                        <svg viewBox="0 0 200 200" aria-hidden="true">
                            <g class="roam-land">
                                <path class="roam-land-highlight" d="M19 41 29 29 43 26 49 19 63 22 70 31 66 39 74 45 67 53 60 55 56 66 49 69 44 61 36 58 31 50 24 48Z"/>
                                <path d="m52 71 8 5 5 12-4 12-5 12-3 15-6-4-2-12-5-10 2-12-5-8 7-6Z"/>
                                <path d="m82 36 7-7 12 2 4 7-8 5-9-1Z"/>
                                <path class="roam-land-highlight" d="m91 54 9-5 10 3 8 10-2 12-6 7-2 13-7 13-6 16-7-2-3-13-7-10 1-13-5-11 5-12Z"/>
                                <path d="m105 38 11-9 19-4 13 5 14-1 16 8-3 8-13 2-8 7-12-3-8 8-10-4-6-9-11 1-5-5Z"/>
                                <path class="roam-land-highlight" d="m145 81 10 2 7 8-4 8-8-2-6-7Z"/>
                                <path d="m160 117 12-2 10 8-5 10-13 2-7-8Z"/>
                            </g>
                            <g class="roam-grid">
                                <path d="M0 52 Q100 67 200 52M0 100 Q100 112 200 100M0 148 Q100 133 200 148"/>
                                <path d="M45 0 C76 52 76 148 45 200M100 0 C116 52 116 148 100 200M155 0 C124 52 124 148 155 200"/>
                            </g>
                        </svg>
                    </div>
                    <div class="roam-globe-shade"></div>
                </div>
                <div class="roam-float-card roam-float-card-one">
                    <span class="roam-float-icon">⌖</span>
                    <span><strong>Discover places</strong><small>Local highlights</small></span>
                </div>
                <div class="roam-float-card roam-float-card-two">
                    <span class="roam-float-icon">✦</span>
                    <span><strong>Your itinerary</strong><small>Made around you</small></span>
                </div>
                <div class="roam-pin roam-pin-one">🌍 Explore the world</div>
                <div class="roam-pin roam-pin-two">✈️ Your next adventure</div>
            </div>
            <div class="roam-hero-glow"></div>
        </header>
        """
    )
    gr.Markdown(
        "### A little help for every part of the trip",
        elem_classes=["roam-section-title"],
    )
    gr.Markdown(
        (
            "Live web search is available when a visitor supplies a Tavily key in the startup dialog. "
            "A deployment-level TAVILY_API_KEY secret can also be used as a fallback."
        ),
        elem_classes=["roam-status"],
    )
    gr.HTML(
        """
        <section class="roam-services" aria-label="Travel assistant features">
            <article class="roam-service">
                <span class="roam-service-icon" aria-hidden="true">01</span>
                <strong>Find your places</strong>
                <span>Explore mapped stays, restaurants, attractions, and airports.</span>
            </article>
            <article class="roam-service">
                <span class="roam-service-icon" aria-hidden="true">02</span>
                <strong>Get current context</strong>
                <span>Search for recent travel advisories and destination details.</span>
            </article>
            <article class="roam-service">
                <span class="roam-service-icon" aria-hidden="true">03</span>
                <strong>Plan the journey</strong>
                <span>Get a flight-search link and shape a trip around your dates.</span>
            </article>
        </section>
        """
    )
    with gr.Column(elem_id="trip-planner", elem_classes=["roam-chat"]):
        trip_textbox = gr.Textbox(
            placeholder="Where would you like to go? Tell me your dates, budget, or interests...",
            container=False,
            scale=7,
            submit_btn="Plan my trip",
            stop_btn="Stop",
        )
        gr.ChatInterface(
            fn=travel_agent_chat,
            chatbot=gr.Chatbot(
                height=480,
                label="Your travel conversation",
                buttons=["copy"],
                layout="bubble",
                render_markdown=True,
                placeholder="Your trip ideas and recommendations will appear here.",
            ),
            textbox=trip_textbox,
            cache_examples=False,
            additional_inputs=[openai_api_key_state, tavily_api_key_state],
        )
        with gr.Row(elem_classes=["roam-examples"]):
            for prompt in (
                "Plan a 3-day food and culture trip to Paris.",
                "Find mapped hotels, attractions, and airports near Tokyo.",
                "What travel advisories should I check before visiting New York?",
                "Find a flight search link from Toronto to Lisbon for next month.",
            ):
                prompt_button = gr.Button(prompt, variant="secondary")
                prompt_button.click(
                    fn=_prompt_handler(prompt),
                    inputs=None,
                    outputs=[trip_textbox],
                    queue=False,
                    api_name=False,
                )
    gr.HTML(
        """
        <footer class="roam-note">
            OpenStreetMap results can be incomplete. Hotel listings do not confirm
            availability or prices, and flight links do not quote live fares.
            Always verify details with travel providers. Map data © OpenStreetMap
            contributors. OpenAI and web-search requests may incur account charges.
        </footer>
        """
    )


def launch_app() -> None:
    """Launch the Gradio app using platform-provided host and port settings."""
    demo.launch(
        theme=theme,
        css=APP_CSS,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(
            os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))
        ),
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
        inbrowser=os.environ.get("GRADIO_INBROWSER", "false").lower() == "true",
        # Hugging Face Spaces works reliably with Gradio's built-in server;
        # disable the experimental Node.js SSR sidecar on hosted deployments.
        ssr_mode=False,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    launch_app()
