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
    assert "\\nspaces" not in requirements


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
