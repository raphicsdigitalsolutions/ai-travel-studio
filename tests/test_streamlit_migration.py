from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_entrypoint_exists_and_uses_streamlit():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "import streamlit as st" in source
    assert "st.chat_input" in source


def test_streamlit_requirements_include_streamlit_and_not_gradio():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "streamlit" in requirements
    assert "gradio" not in requirements
    assert "spaces" not in requirements


def test_streamlit_config_sets_server_defaults():
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'headless = true' in config
    assert 'port = 8501' in config


def test_legacy_hugging_face_deployment_is_not_overwritten():
    workflow = (ROOT / ".github" / "workflows" / "deploy-huggingface.yml").read_text(
        encoding="utf-8"
    )
    assert "Hugging Face" in workflow
    assert "HF_TOKEN" in workflow
