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

Set `TAVILY_API_KEY` in your hosting provider's private environment/secrets panel to
enable current web search. Without it, the app still starts and supports AI planning,
OpenStreetMap discovery, and flight-search links; the interface indicates that live
web search is disabled.

Optional model setting:

- `OPENAI_MODEL` — defaults to `gpt-4o-mini`.

## Run locally on Windows

1. Install Python 3.10 or newer.
2. Open PowerShell in this repository folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Optional: create a `.env` file in the project folder:

   ```env
   TAVILY_API_KEY=your_tavily_key
   OPENAI_MODEL=gpt-4o-mini
   ```

6. Start the app:

   ```powershell
   python travel_agent.py
   ```

7. Open `http://127.0.0.1:7860` and enter your OpenAI key in the startup modal.

The server defaults to `0.0.0.0` so a hosting platform can reach it. For local use,
the app is still opened on `127.0.0.1` by your browser.

## Deploy online

GitHub stores the code; it does not host the running Gradio process. Choose a Python
app host and add the repository as a deployment source.

### Option A — Hugging Face Spaces (straightforward for Gradio)

1. Sign in at [Hugging Face](https://huggingface.co/) and create a new Space.
2. Choose **Gradio** as the SDK and select an appropriate hardware tier.
3. Link/import this GitHub repository, or push the repository contents to the Space.
4. In the Space, open **Settings → Variables and secrets**.
5. Add `TAVILY_API_KEY` as a secret (optional, but required for live web search).
6. Keep `OPENAI_MODEL` as a variable if you want to override the default.
7. Make sure the Space uses `travel_agent.py` as its app entry file. If the Space
   expects `app.py`, either rename the entry file or configure the SDK entry point
   to `travel_agent.py` where the platform supports it.
8. The app binds to the platform-provided `PORT` and `0.0.0.0`. Once the build
   completes, open the Space URL and test the OpenAI key modal and a simple prompt.

### Option B — Render / Railway / another Python host

1. Create a new Python web service from this GitHub repository.
2. Build/install command: `pip install -r requirements.txt`.
3. Start command: `python travel_agent.py`.
4. Add `TAVILY_API_KEY` (optional) and `OPENAI_MODEL` in the provider's environment
   settings. Never paste secrets into the start command or commit them.
5. Ensure the host injects `PORT`; this app reads it automatically and binds to
   `0.0.0.0`.
6. Enable HTTPS on the provider's domain, then test the public URL.

For hosts that require a fixed port, set `GRADIO_SERVER_PORT`. You can also set
`GRADIO_SERVER_NAME` or `GRADIO_SHARE=true` as environment variables. Avoid
Gradio share links for production; use the host's managed HTTPS domain.

## Data and service limitations

- OpenStreetMap data can be incomplete or outdated. Verify important places and
  addresses with the provider before travelling.
- Nominatim and Overpass are public community services with usage policies and
  rate limits. The app caches geocoding lookups and spaces requests, but a public
  deployment with many visitors may need a hosted geocoding/map-data provider.
- OpenStreetMap does not confirm accommodation availability, hotel pricing, flight
  schedules, or live fares.
- OpenAI and Tavily usage can incur charges on the respective accounts.
- The visitor-supplied OpenAI key is used for model calls; Tavily is a server-side
  deployment secret shared by the app deployment.

## Project files

- `travel_agent.py` — Gradio interface, API-key gate, LangGraph workflow, and tools.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes local secrets, virtual environments, and cache files.

## License and attribution

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
Review the terms of all third-party APIs and hosting providers before publishing.
