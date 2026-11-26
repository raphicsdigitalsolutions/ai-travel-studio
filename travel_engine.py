"""Reusable AI travel-planning engine for the Streamlit Roamly app."""

from __future__ import annotations

import operator
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

OSM_USER_AGENT = "RoamlyTravelStudio/2.0 (AI travel-planning app)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = (
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
_geocode_cache: dict[str, tuple[float, float]] = {}
_nominatim_lock = threading.Lock()
_last_nominatim_request = 0.0


def _geocode(place: str) -> tuple[float, float]:
    """Resolve a place with Nominatim and respect its one-request-per-second policy."""
    global _last_nominatim_request

    normalized = place.strip()
    if not normalized:
        raise ValueError("Please provide a city or place name.")

    cache_key = normalized.casefold()
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
            params={"q": normalized, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": OSM_USER_AGENT},
            timeout=20,
        )
        _last_nominatim_request = time.monotonic()
        response.raise_for_status()
        results = response.json()

    if not results:
        raise ValueError(f"OpenStreetMap could not find a location for {normalized!r}.")

    coordinates = (float(results[0]["lat"]), float(results[0]["lon"]))
    _geocode_cache[cache_key] = coordinates
    return coordinates


def _query_nearby(place: str, tag_filters: list[str], radius_km: int = 5) -> list[dict]:
    if not 1 <= radius_km <= 50:
        raise ValueError("Search radius must be between 1 and 50 km.")

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
                timeout=25,
            )
            response.raise_for_status()
            return response.json().get("elements", [])
        except requests.RequestException as error:
            last_error = error

    map_url = "https://www.openstreetmap.org/search?query=" + quote_plus(place)
    raise RuntimeError(
        "OpenStreetMap's place-search service is temporarily unavailable. "
        f"Try again later or search directly: {map_url}"
    ) from last_error


def _format_places(elements: list[dict], city: str, limit: int = 5) -> str:
    if not 1 <= limit <= 10:
        raise ValueError("The maximum results must be between 1 and 10.")

    lines: list[str] = []
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
        + "\n\nMap data © OpenStreetMap contributors "
        "(https://www.openstreetmap.org/copyright)."
    )


@tool
def find_airports_tool(city: str, radius_km: int = 50, max_results: int = 5) -> str:
    """Find mapped airports near a city; IATA codes may be missing from map data."""
    elements = _query_nearby(city, ['["aeroway"="aerodrome"]'], radius_km)
    return _format_places(elements, city, max_results)


@tool
def search_hotels_tool(city: str, radius_km: int = 5, max_results: int = 5) -> str:
    """Find mapped hotels and similar accommodation; prices and availability are not supplied."""
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
    """Find mapped restaurants, cafes, attractions, museums, and parks near a city."""
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
    """Create a Google Flights search link; this app does not provide live fares."""
    if adults < 1:
        raise ValueError("The number of adults must be at least one.")

    query = (
        f"Flights from {origin_city} to {destination_city} on {departure_date} "
        f"for {adults} adults"
    )
    if return_date:
        query += f", returning {return_date}"

    url = "https://www.google.com/travel/flights?q=" + quote_plus(query)
    return (
        "I cannot verify live flight schedules or fares from the map data. "
        f"Search current options here: {url}"
    )


@tool
def get_current_date_tool() -> str:
    """Return today's date for travel plans."""
    from datetime import date
    return date.today().isoformat()


@tool
def search_web_tool(query: str, config: RunnableConfig) -> str:
    """Search the live web for current travel information, advisories, and destination details."""
    configurable = config.get("configurable", {})
    api_key = configurable.get("tavily_api_key") or os.environ.get("TAVILY_API_KEY", "")
    if not isinstance(api_key, str) or not api_key.strip():
        return (
            "Live web search is not configured. The user can add a Tavily API key "
            "in the app sidebar, or continue with OpenStreetMap place discovery."
        )

    try:
        result = TavilySearch(max_results=5, tavily_api_key=api_key.strip()).invoke(
            {"query": query}
        )
        return str(result)
    except Exception as error:
        return (
            "Live web search failed. Check the Tavily key and service availability, "
            f"then try again. Error type: {type(error).__name__}"
        )


TOOLS = [
    search_web_tool,
    find_airports_tool,
    search_hotels_tool,
    search_local_places_tool,
    search_flights_tool,
    get_current_date_tool,
]


class TravelState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def call_model(state: TravelState, config: RunnableConfig) -> dict[str, list[AIMessage]]:
    api_key = config.get("configurable", {}).get("openai_api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError("Add your OpenAI API key in the sidebar to start planning.")

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
    )
    response = model.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [response]}


def route_after_model(state: TravelState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(TravelState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", route_after_model, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
travel_agent = workflow.compile()

SYSTEM_PROMPT = """You are Roamly, a practical and thoughtful AI travel-planning assistant.
Use search_web_tool for current travel advisories, recent information, and destination
details. If there is no Tavily key or the search fails, be transparent and never claim
that a live search succeeded. Use OpenStreetMap tools for mapped airports, hotels,
restaurants, and attractions. OSM data may be incomplete; hotel listings do not mean
availability or prices are known. OSM does not provide flight schedules or live fares:
create a flight-search link and never invent prices, routes, or availability. Ask a
short follow-up if an essential trip detail is missing. Clearly distinguish sourced
facts from suggestions and include map links. Prefer practical, well-structured
itineraries with sensible travel pacing."""


def run_travel_agent(
    conversation: list[BaseMessage],
    openai_api_key: str,
    tavily_api_key: str = "",
) -> tuple[str, list[str]]:
    """Run one conversation turn and return the final assistant response and used tools."""
    if not openai_api_key or not openai_api_key.strip():
        raise ValueError("Add your OpenAI API key in the sidebar.")

    input_messages = [SystemMessage(content=SYSTEM_PROMPT), *conversation]
    final_response = ""
    used_tools: list[str] = []

    for update in travel_agent.stream(
        {"messages": input_messages},
        {
            "recursion_limit": 15,
            "configurable": {
                "openai_api_key": openai_api_key.strip(),
                "tavily_api_key": tavily_api_key.strip(),
            },
        },
        stream_mode="updates",
    ):
        for node_update in update.values():
            for message in node_update.get("messages", []):
                if isinstance(message, ToolMessage):
                    if message.name not in used_tools:
                        used_tools.append(message.name)
                elif isinstance(message, AIMessage) and message.content:
                    final_response = str(message.content)

    if not final_response:
        final_response = "I couldn't generate a response. Please try again."
    return final_response, used_tools
