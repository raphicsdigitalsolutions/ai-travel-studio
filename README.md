---
title: Roamly - 3D AI Travel Studio
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
---

# Roamly — 3D AI Travel Studio

Roamly is a Python + Gradio travel-planning assistant powered by OpenAI, LangGraph,
OpenStreetMap, and optional Tavily web search. The interface uses a teal/mint visual
system with a floating 3D globe, orbit rings, travel pins, layered cards, responsive
layouts, and a secure OpenAI API-key gate.

## What it can do

- Plan trips and answer travel-planning questions through an AI chat.
- Find mapped airports, hotels, restaurants, and attractions with OpenStreetMap.
- Generate Google Flights search links. Flight schedules and fares are not supplied
  by OpenStreetMap, so the app does not invent live prices.
- Search current travel information when a Tavily key is configured.
- Run in light or dark Gradio theme mode and adapt to mobile screens.

## API keys and security

### OpenAI key (entered by each visitor)

The app displays a password-style modal when a session starts. Visitors enter their
own OpenAI API key and confirm that they trust the deployment before the app verifies
it with OpenAI. The key is held in that Gradio session's server-side state and is not
written to a file or browser local storage.

**Important:** a visitor's key is sent to the server hosting this app. Only enter it
on a deployment you control or trust. Use HTTPS for public deployments. Do not add
API keys to Git, screenshots, logs, or README files.

### Tavily key (optional server secret)

Set `TAVILY_API_KEY` in the Space's Settings → Variables and secrets to enable current
web search. Without it, the app still starts and supports AI planning, OpenStreetMap
discovery, and flight-search links; the interface indicates that live web search is
disabled.

Optional model setting:

- `OPENAI_MODEL` — defaults to `gpt-4o-mini`.

## Deploy on Hugging Face Spaces

This repository's Space entry point is `app.py`. Upload `app.py`,
`travel_agent.py`, `requirements.txt`, and this README to the Space repository.
Choose **Gradio** SDK and **CPU Basic** hardware. The visitor enters their own
OpenAI API key through the app's startup modal.

## Run locally on Windows

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   python app.py
   ```

5. Open `http://127.0.0.1:7860`.

## Data and service limitations

- OpenStreetMap data can be incomplete or outdated. Verify important places and
  addresses with the provider before travelling.
- Nominatim and Overpass are public community services with usage policies and
  rate limits.
- OpenStreetMap does not confirm accommodation availability, hotel pricing, flight
  schedules, or live fares.
- OpenAI and Tavily usage can incur charges on the respective accounts.

## Project files

- `app.py` — Gradio deployment entry point.
- `travel_agent.py` — Gradio interface, API-key gate, LangGraph workflow, tools, and launcher.
- `requirements.txt` — Python dependencies.

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
