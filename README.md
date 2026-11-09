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

### Tavily key (optional visitor key or server secret)

The startup popup asks for:
- **OpenAI API key** — required for AI trip planning.
- **Tavily API key** — optional; enables live web search for current travel information.

You can leave the Tavily field blank. OpenStreetMap place discovery and AI trip
planning still work, but live web search will not be available unless a key is
provided.

**Get a Tavily key for free:**
1. Open [Tavily](https://tavily.com/) and create a free account (or sign in).
2. Open the [Tavily dashboard](https://app.tavily.com/).
3. Create or copy your API key from the API key section.
4. Paste it into the optional Tavily field in Roamly's startup popup.

Tavily offers a free usage tier; available credits and limits can change, so check
the current plan details in your Tavily dashboard. Never publish your key or paste
it into GitHub source code.

For an owner-managed deployment, you can instead add `TAVILY_API_KEY` under the
Hugging Face Space's **Settings → Variables and secrets**. This server secret is
used as a fallback when a visitor does not enter a Tavily key. Do not add it to
the repository or README.

Optional model setting:

- `OPENAI_MODEL` — defaults to `gpt-4o-mini`.

## Automatic deployment from GitHub

Every push to the GitHub `main` branch triggers
`.github/workflows/deploy-huggingface.yml`, which syncs the repository to this
Space:

`attaurrehmansiddiqui/roamly-travel-studio`

One-time setup:

1. On Hugging Face, open **Settings → Access Tokens** and create a token with
   **Write** access (or a fine-grained token with write access to this Space).
2. In GitHub, open this repository's **Settings → Secrets and variables → Actions**.
3. Select **New repository secret**.
4. Set the name to `HF_TOKEN` and paste the Hugging Face token as its value.
5. Save. Then open GitHub **Actions** and run **Deploy Roamly to Hugging Face Spaces**
   once with **Run workflow**, or push a new commit to `main`.

Do not put the token in source files, README content, or commits. If a workflow
fails, inspect its GitHub Actions run logs.

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
