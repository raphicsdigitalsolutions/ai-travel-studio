from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_entrypoint_wires_chat_to_travel_engine():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "import streamlit as st" in source
    assert "st.chat_input" in source
    assert "from travel_engine import run_travel_agent" in source


def test_requirements_use_streamlit_without_gradio_or_spaces():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "streamlit" in requirements
    assert "langgraph" in requirements
    assert "gradio" not in requirements
    assert "spaces" not in requirements


def test_streamlit_config_sets_server_and_theme_defaults():
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "headless = true" in config
    assert "port = 8501" in config
    assert 'primaryColor = "#0B9E8D"' in config


def test_travel_engine_contains_agent_and_openstreetmap_tools():
    source = (ROOT / "travel_engine.py").read_text(encoding="utf-8")
    assert "def run_travel_agent(" in source
    assert "def _format_places(" in source
    assert "find_airports_tool" in source
    assert "search_hotels_tool" in source


def test_readme_documents_streamlit_cloud_deployment():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Streamlit Community Cloud" in readme
    assert "app.py" in readme
    assert "OPENAI_API_KEY" in readme

def test_place_formatter_returns_openstreetmap_links_and_disclaimer():
    from travel_engine import _format_places

    result = _format_places(
        [{"lat": 31.5204, "lon": 74.3587, "tags": {"name": "Test Place", "amenity": "cafe"}}],
        "Lahore",
    )
    assert "[Test Place](https://www.openstreetmap.org/" in result
    assert "OpenStreetMap contributors" in result


def test_flight_tool_never_claims_to_provide_live_fares():
    from travel_engine import search_flights_tool

    result = search_flights_tool.invoke(
        {
            "origin_city": "Lahore",
            "destination_city": "Istanbul",
            "departure_date": "2027-01-15",
            "adults": 1,
        }
    )
    assert "google.com/travel/flights" in result
    assert "cannot verify live flight schedules or fares" in result


def test_first_visit_opens_api_key_dialog_and_sidebar_can_reopen_it():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '@st.dialog("Connect Roamly to your AI", width="small"' in source
    assert 'st.session_state.show_key_dialog = not bool(st.session_state.openai_key)' in source
    assert 'st.button("Connect / manage API keys"' in source
    assert 'st.session_state.show_key_dialog = False' in source


def test_api_key_dialog_requires_explicit_trust_before_saving_visitor_keys():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    dialog = source.split('@st.dialog("Connect Roamly to your AI", width="small"', 1)[1].split("\nwith st.sidebar:", 1)[0]
    assert 'st.checkbox("I trust this deployment with the API keys I enter.")' in dialog
    assert 'elif not trust:' in dialog
    assert 'st.session_state.openai_key = openai_input.strip()' in dialog
