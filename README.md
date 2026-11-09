# LangGraph Travel Assistant

This local Gradio app combines:

- OpenStreetMap (Nominatim and Overpass) for mapped hotels, restaurants,
  attractions, and airports.
- Tavily for current web search, including travel advisories.
- OpenAI and LangGraph for tool-using chat.
- Google Flights search links for flight requests.

OpenStreetMap does not provide live flight schedules/fares or hotel availability
and prices. Always verify listings and travel details with the provider.

## API keys and deployment

- The app asks for an OpenAI API key in a password-style prompt for each new
  session. It verifies the key, keeps it in that session's server-side memory,
  and does not write it to a file or browser storage.
- The key is sent to the app server to make OpenAI requests. Only enter it on a
  deployment you trust, and serve the public app over HTTPS.
- `TAVILY_API_KEY` is still required by the web-search feature. Set it as a
  private environment secret in your hosting provider; do not commit it or the
  local `.env` file to GitHub.
- GitHub stores the source code but does not run this Gradio app by itself.
  Deploy it to a Python app host and configure `TAVILY_API_KEY` there.

## Setup on Windows

1. Keep `TAVILY_API_KEY` in the `.env` file beside `travel_agent.py`. The
   OpenAI key is entered in the app prompt. Do not share or commit either key.
2. From this folder, create and activate a virtual environment, then install
   the requirements:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   python travel_agent.py
   ```

The app opens on `http://127.0.0.1:7860`. OpenAI and Tavily requests may incur
charges according to your accounts. OpenStreetMap's public services are
rate-limited; Nominatim requests are cached and spaced, and map results include
OpenStreetMap attribution.
