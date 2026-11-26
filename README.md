# Roamly — 3D AI Travel Studio

Roamly is a Streamlit-based AI travel-planning studio powered by OpenAI, LangGraph, OpenStreetMap, and optional Tavily web search. The UI uses a teal/mint visual system, animated 3D-style globe, orbit rings, floating travel cards, responsive layout, and a private API-key panel.

## Features

- AI trip planning and conversational itinerary assistance.
- OpenStreetMap discovery for airports, hotels, restaurants, cafes, and attractions.
- Google Flights search links (the app does not claim to have live schedules or fares).
- Optional Tavily web search for current travel information and advisories.
- Session-based chat history and API-key handling.

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Select **Create app**.
3. Choose this repository: `raphicsdigitalsolutions/ai-travel-studio`.
4. Set branch to `feat/streamlit-deployment` (or `main` after the migration is merged).
5. Set the app file to `app.py`.
6. Under **Advanced settings → Secrets**, optionally add owner-managed keys in TOML format:

   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   TAVILY_API_KEY = "your-tavily-api-key"
   OPENAI_MODEL = "gpt-4o-mini"
   ```

7. Click **Deploy**. Streamlit installs packages from `requirements.txt`.

You can also deploy without owner-managed secrets: each visitor can enter their own OpenAI key in the sidebar, and optionally a Tavily key. Only do this on a deployment you control or trust.

## Run locally on Windows

Use Python 3.11 or newer:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

For local keys, create a `.env` file beside `app.py` (do not commit it):

```dotenv
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
OPENAI_MODEL=gpt-4o-mini
```

## Deploy on Render (optional)

The included `render.yaml` starts Streamlit on Render's assigned port. Add `OPENAI_API_KEY` and optionally `TAVILY_API_KEY` in the Render service environment settings. Streamlit Community Cloud is the primary deployment target.

## API key and privacy notes

- Keys entered in the sidebar are stored in the server-side Streamlit session state and are not written into repository files or browser local storage.
- With owner-managed secrets, the configured server key is used by the app. Do not publish secrets or paste them into GitHub.
- Enter keys only on HTTPS deployments you control or trust. A visitor-entered key is sent to the server hosting the application to make requests to the corresponding provider.
- The app does not verify hotel prices or accommodation availability.
- OpenStreetMap data may be incomplete or outdated. Nominatim and Overpass are public services with usage policies and rate limits.
- OpenAI and Tavily usage may incur charges on the relevant accounts.

## Project structure

- `app.py` — Streamlit UI and session-based chat.
- `travel_engine.py` — LangGraph agent, OpenStreetMap tools, flight search links, and Tavily search.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — Streamlit server and theme defaults.
- `render.yaml` — optional Render deployment blueprint.

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
