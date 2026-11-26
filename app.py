"""Roamly — 3D AI Travel Studio, powered by Streamlit."""

from __future__ import annotations

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from travel_engine import run_travel_agent

st.set_page_config(
    page_title="Roamly | AI Travel Studio",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
      --ink:#102d32; --muted:#55777a; --teal:#0b9e8d;
      --mint:#dff7ed; --line:rgba(18,130,111,.14);
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
      background:
        radial-gradient(ellipse at 7% 4%,rgba(21,181,145,.16),transparent 34rem),
        radial-gradient(ellipse at 92% 46%,rgba(22,135,120,.10),transparent 34rem),
        linear-gradient(155deg,#f3fbf7 0%,#eaf7f1 48%,#f4faf7 100%);
      color:var(--ink);
    }
    [data-testid="stHeader"] { background:rgba(244,251,247,.72); }
    [data-testid="stSidebar"] {
      background:linear-gradient(180deg,#e1f4eb 0%,#f1faf5 100%);
      border-right:1px solid rgba(17,122,101,.12);
    }
    [data-testid="stSidebar"] > div { padding-top:1.4rem; }
    .block-container { max-width:1320px; padding-top:2rem; padding-bottom:4rem; }
    .roam-topline {
      display:flex; align-items:center; gap:9px; color:#218675;
      font-size:11px; font-weight:800; letter-spacing:.19em; text-transform:uppercase;
      margin:0 0 13px;
    }
    .roam-dot {
      width:8px;height:8px;border-radius:50%;background:#20b68e;
      box-shadow:0 0 0 5px rgba(32,182,142,.12),0 0 18px rgba(32,182,142,.65);
    }
    .roam-hero {
      position:relative; display:flex; align-items:center; justify-content:space-between;
      gap:30px; min-height:390px; padding:52px 58px; overflow:hidden;
      border:1px solid rgba(255,255,255,.9); border-radius:32px;
      background:
        radial-gradient(circle at 78% 45%,rgba(43,204,157,.23),transparent 19rem),
        linear-gradient(125deg,#fafffc 0%,#e8f8f0 48%,#d8f1e8 100%);
      box-shadow:0 28px 70px rgba(25,104,86,.11),inset 0 1px 0 #fff;
    }
    .roam-hero:before,.roam-hero:after {
      content:"";position:absolute;border:1px solid rgba(18,154,124,.11);border-radius:50%;
      pointer-events:none;
    }
    .roam-hero:before { width:480px;height:480px;right:4%;top:-72px;box-shadow:0 0 0 28px rgba(18,154,124,.025),0 0 0 60px rgba(18,154,124,.02); }
    .roam-hero:after { width:330px;height:330px;right:10%;top:3px; }
    .roam-copy { position:relative; z-index:2; max-width:610px; }
    .roam-copy h1 {
      font-family:'Manrope',sans-serif; font-size:clamp(2.6rem,5.2vw,4.7rem);
      line-height:1.02; letter-spacing:-.065em; font-weight:800;
      color:#10383a; margin:0 0 18px;
    }
    .roam-copy p { max-width:570px; color:#597779; font-size:16px; line-height:1.75; margin:0; }
    .roam-tags { display:flex; flex-wrap:wrap; gap:9px; margin-top:25px; }
    .roam-tags span {
      border:1px solid rgba(17,139,111,.17);border-radius:999px;padding:8px 12px;
      background:rgba(255,255,255,.64);color:#286d63;font-size:11px;font-weight:700;
    }
    .globe-stage {
      position:relative;z-index:1;width:350px;height:300px;min-width:270px;
      display:grid;place-items:center;perspective:1000px;
    }
    .globe {
      position:relative;width:205px;height:205px;border-radius:50%;
      background:
        radial-gradient(circle at 31% 25%,rgba(255,255,255,.95),transparent 17%),
        radial-gradient(circle at 36% 35%,#67e4b8 0%,#1dbb94 40%,#087d70 78%,#075b59 100%);
      box-shadow:inset -22px -22px 42px rgba(1,50,52,.35),inset 9px 8px 24px rgba(255,255,255,.42),
        0 32px 55px rgba(4,93,78,.24),0 0 55px rgba(27,199,151,.22);
      transform:rotateX(13deg) rotateY(-18deg); animation:globeFloat 7s ease-in-out infinite alternate;
      overflow:hidden;
    }
    .globe:before {
      content:"";position:absolute;inset:9px;border:1px solid rgba(222,255,239,.25);border-radius:50%;
      background:
        radial-gradient(ellipse at 33% 30%,rgba(244,255,249,.42) 0 5%,transparent 6%),
        radial-gradient(ellipse at 65% 42%,rgba(226,255,238,.35) 0 9%,transparent 10%),
        radial-gradient(ellipse at 46% 68%,rgba(226,255,238,.28) 0 7%,transparent 8%);
      box-shadow:inset 0 0 24px rgba(255,255,255,.14);
    }
    .globe:after {
      content:"";position:absolute;inset:0;border-radius:50%;
      background:linear-gradient(115deg,transparent 22%,rgba(255,255,255,.28) 42%,transparent 57%);
      transform:translateX(-100%);animation:globeShine 6s ease-in-out infinite;
    }
    .orbit {
      position:absolute;width:325px;height:126px;border:1.5px solid rgba(10,139,119,.36);
      border-radius:50%;transform:rotate(-25deg) rotateX(66deg);
      box-shadow:0 0 18px rgba(14,165,135,.11);animation:orbit 18s linear infinite;
    }
    .orbit.two { width:290px;height:165px;transform:rotate(55deg) rotateX(68deg);animation-duration:24s;animation-direction:reverse; }
    .orbit:after {
      content:"";position:absolute;top:7px;left:18%;width:10px;height:10px;border-radius:50%;
      background:#fff;box-shadow:0 0 0 5px rgba(22,177,139,.13),0 0 18px #27c99b;
    }
    .float-chip {
      position:absolute;padding:11px 14px;border:1px solid rgba(255,255,255,.88);
      border-radius:15px;background:rgba(255,255,255,.78);backdrop-filter:blur(12px);
      box-shadow:0 12px 26px rgba(9,83,72,.12);font-size:11px;color:#1a5e58;font-weight:800;
    }
    .chip-one { top:28px;left:0;transform:rotate(-5deg); }
    .chip-two { bottom:25px;right:-3px;transform:rotate(4deg); }
    @keyframes globeFloat { from{transform:rotateX(12deg) rotateY(-15deg) translateY(0)} to{transform:rotateX(-8deg) rotateY(14deg) translateY(-11px)} }
    @keyframes globeShine { 0%,35%{transform:translateX(-110%)} 75%,100%{transform:translateX(110%)} }
    @keyframes orbit { from{rotate:0deg} to{rotate:360deg} }
    .section-kicker { margin:30px 0 7px;color:#258b78;font-size:11px;font-weight:800;letter-spacing:.17em;text-transform:uppercase; }
    .section-title { margin:0 0 15px;color:#153e40;font-family:'Manrope',sans-serif;font-size:24px;font-weight:800;letter-spacing:-.04em; }
    .feature-card {
      min-height:138px;padding:20px 21px;border:1px solid rgba(255,255,255,.95);
      border-radius:20px;background:linear-gradient(145deg,rgba(255,255,255,.88),rgba(230,248,239,.88));
      box-shadow:0 12px 26px rgba(18,100,83,.07),inset 0 1px 0 #fff;
    }
    .feature-num { color:#28a98a;font-size:11px;font-weight:900;letter-spacing:.12em; }
    .feature-card h3 { margin:10px 0 6px;color:#19484a;font-size:15px;font-weight:800; }
    .feature-card p { margin:0;color:#668284;font-size:12px;line-height:1.6; }
    .sidebar-brand { font-family:'Manrope',sans-serif;font-weight:900;font-size:23px;letter-spacing:-.05em;color:#14584f; }
    .sidebar-sub { margin-top:0;color:#688780;font-size:11px;letter-spacing:.12em;text-transform:uppercase; }
    .stButton > button {
      border:1px solid rgba(14,151,121,.19);border-radius:13px;
      background:linear-gradient(180deg,#22c99c,#07836d);color:white;font-weight:800;
      box-shadow:0 5px 0 #086b5a,0 11px 22px rgba(10,144,115,.16);
      transition:transform .18s ease,box-shadow .18s ease;
    }
    .stButton > button:hover { color:white;border-color:#0a9278;transform:translateY(-2px);box-shadow:0 7px 0 #086b5a,0 15px 25px rgba(10,144,115,.2); }
    [data-testid="stChatMessage"] { border:1px solid rgba(255,255,255,.8);border-radius:17px;background:rgba(255,255,255,.6); }
    [data-testid="stChatInput"] { border-radius:16px; }
    .note { color:#718c8b;font-size:11px;line-height:1.7; }
    .connect-dialog-copy { text-align:center;padding:3px 4px 15px; }
    .connect-orb {
      width:54px;height:54px;margin:0 auto 12px;display:grid;place-items:center;
      border-radius:18px;color:white;font-size:25px;
      background:linear-gradient(145deg,#55dfb2,#07836d);
      box-shadow:0 10px 24px rgba(10,144,115,.22),inset 0 1px 0 rgba(255,255,255,.65);
    }
    .connect-dialog-copy h2 {
      margin:0 0 8px;color:#153e40;font-family:'Manrope',sans-serif;
      font-size:23px;font-weight:800;letter-spacing:-.045em;
    }
    .connect-dialog-copy p { margin:0;color:#668284;font-size:13px;line-height:1.65; }
    .destination-card {
      min-height:184px;padding:19px 20px 17px;border-radius:21px;
      border:1px solid rgba(255,255,255,.96);
      background:radial-gradient(circle at 100% 0,rgba(48,203,158,.18),transparent 9rem),
        linear-gradient(145deg,rgba(255,255,255,.92),rgba(225,246,236,.9));
      box-shadow:0 13px 30px rgba(18,100,83,.075),inset 0 1px 0 #fff;
      transition:transform .2s ease,box-shadow .2s ease;
    }
    .destination-card:hover { transform:translateY(-3px);box-shadow:0 18px 34px rgba(18,100,83,.11); }
    .destination-symbol {
      width:34px;height:34px;display:grid;place-items:center;border-radius:11px;
      color:#07836d;background:rgba(31,190,145,.13);font-size:18px;font-weight:900;
    }
    .destination-mood { margin-top:13px;color:#29927d;font-size:10px;font-weight:800;letter-spacing:.11em;text-transform:uppercase; }
    .destination-card h3 { margin:4px 0 5px;color:#174548;font-family:'Manrope',sans-serif;font-size:19px;font-weight:800;letter-spacing:-.035em; }
    .destination-card p { margin:0;color:#688382;font-size:12px;line-height:1.55; }
    [data-testid="stDialog"] [data-testid="stForm"] { border:0;padding:0; }
    [data-testid="stDialog"] [data-testid="stFormSubmitButton"] button { min-height:45px; }
    @media(max-width:850px) {
      .roam-hero { padding:34px 27px;min-height:0;flex-direction:column;align-items:flex-start; }
      .globe-stage { width:100%;height:250px;min-width:0; }
      .globe { width:170px;height:170px; }
      .orbit { width:270px; }
      .roam-copy p { font-size:14px; }
    }
    @media(prefers-reduced-motion:reduce) { *,*:before,*:after {animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;} }
    </style>
    """,
    unsafe_allow_html=True,
)

def _secret(name: str, fallback: str = "") -> str:
    """Read a deployment secret without exposing it in the page."""
    value = os.environ.get(name, "")
    try:
        value = value or st.secrets.get(name, fallback)
    except Exception:
        pass
    return str(value or "").strip()


os.environ["OPENAI_MODEL"] = _secret("OPENAI_MODEL", "gpt-4o-mini")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "openai_key" not in st.session_state:
    st.session_state.openai_key = _secret("OPENAI_API_KEY")
if "tavily_key" not in st.session_state:
    st.session_state.tavily_key = _secret("TAVILY_API_KEY")
# Bump this when the first-run connection flow changes. Streamlit preserves
# session_state across reruns and deployments, so older dismissed dialogs must
# be reconsidered after a UI update.
KEY_DIALOG_VERSION = 2
if st.session_state.get("_key_dialog_version") != KEY_DIALOG_VERSION:
    st.session_state.show_key_dialog = not bool(st.session_state.openai_key)
    st.session_state["_key_dialog_version"] = KEY_DIALOG_VERSION
elif "show_key_dialog" not in st.session_state:
    st.session_state.show_key_dialog = not bool(st.session_state.openai_key)


@st.dialog("Connect Roamly to your AI", width="small")
def api_key_dialog():
    st.markdown(
        """
        <div class="connect-dialog-copy">
          <div class="connect-orb">✦</div>
          <h2>Your journey starts with a connection</h2>
          <p>Add an OpenAI API key to unlock personalised trip planning. A Tavily key is optional for live travel research.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("first_run_api_keys_form"):
        openai_input = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Required to generate AI travel plans.",
        )
        tavily_input = st.text_input(
            "Tavily API key (optional)",
            type="password",
            placeholder="tvly-...",
            help="Enables current web research and travel advisories.",
        )
        trust = st.checkbox("I trust this deployment with the API keys I enter.")
        submitted = st.form_submit_button(
            "Save connection & start planning",
            type="primary",
            use_container_width=True,
        )
        if submitted:
            if not openai_input.strip():
                st.error("An OpenAI API key is required.")
            elif not trust:
                st.error("Confirm that you trust this deployment before sending keys.")
            else:
                st.session_state.openai_key = openai_input.strip()
                st.session_state.tavily_key = tavily_input.strip()
                st.session_state.show_key_dialog = False
                st.rerun()
    if st.button("Explore first — I'll add my key later", use_container_width=True):
        st.session_state.show_key_dialog = False
        st.rerun()


if st.session_state.show_key_dialog and not st.session_state.openai_key:
    api_key_dialog()


with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌍 Roamly</div><div class="sidebar-sub">AI Travel Studio</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("#### Your AI connection")
    st.caption("Use your own API key, or configure owner-managed keys in Streamlit Secrets.")
    with st.form("api_keys_form"):
        openai_input = st.text_input(
            "OpenAI API key",
            value=st.session_state.openai_key,
            type="password",
            placeholder="sk-...",
            help="Required for AI planning. Kept in this Streamlit session only unless configured as an app secret.",
        )
        tavily_input = st.text_input(
            "Tavily API key (optional)",
            value=st.session_state.tavily_key,
            type="password",
            placeholder="tvly-...",
            help="Enables live web search for current travel information.",
        )
        trust = st.checkbox("I trust this deployment with the API keys I enter.")
        save_keys = st.form_submit_button("Save connection", use_container_width=True)
        if save_keys:
            if not openai_input.strip():
                st.error("An OpenAI API key is required.")
            elif not trust:
                st.error("Confirm that you trust this deployment before sending keys.")
            else:
                st.session_state.openai_key = openai_input.strip()
                st.session_state.tavily_key = tavily_input.strip()
                st.success("Keys saved for this session.")
    if st.session_state.openai_key:
        st.success("OpenAI key is set for this session.")
    else:
        st.warning("Add your OpenAI key to start planning.")
    st.divider()
    st.markdown("#### What Roamly can do")
    st.markdown("- Build practical trip itineraries\n- Find mapped hotels and restaurants\n- Discover airports and attractions\n- Create flight-search links\n- Search current travel information with Tavily")
    if st.button("Connect / manage API keys", use_container_width=True):
        st.session_state.show_key_dialog = True
        st.rerun()
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown(
        '<p class="note">Keys are handled by the app server. Do not enter keys on a deployment you do not control or trust.</p>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="roam-topline"><span class="roam-dot"></span> Your next trip starts here</div>
    <section class="roam-hero">
      <div class="roam-copy">
        <h1>Travel more<br>thoughtfully.</h1>
        <p>Meet your personal AI travel companion. Discover places, research destinations, and shape a practical itinerary around your dates, budget, and interests.</p>
        <div class="roam-tags">
          <span>✦ AI trip planner</span><span>⌖ Open map discovery</span><span>↗ Optional live search</span>
        </div>
      </div>
      <div class="globe-stage" aria-hidden="true">
        <div class="orbit"></div><div class="orbit two"></div>
        <div class="globe"></div>
        <div class="float-chip chip-one">⌖ Discover places</div>
        <div class="float-chip chip-two">✦ Your itinerary</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Plan beyond the ordinary</div><div class="section-title">Everything for a more considered journey</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
features = [
    ("01", "Discover destinations", "Explore mapped hotels, restaurants, attractions, and airports."),
    ("02", "Get current context", "Research travel advisories and recent destination information."),
    ("03", "Shape your itinerary", "Build a trip plan and open flight searches for current options."),
]
for column, (number, title, description) in zip((c1, c2, c3), features):
    with column:
        st.markdown(
            f'<div class="feature-card"><div class="feature-num">{number}</div><h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="section-kicker">A little inspiration</div><div class="section-title">Pick a mood. We’ll shape the journey.</div>',
    unsafe_allow_html=True,
)
destination_cols = st.columns(3, gap="medium")
destinations = [
    ("✦", "Culture & city lights", "Istanbul", "Historic streets, waterfront views, and local food.", "Plan a 4-day culture and food trip to Istanbul."),
    ("⌁", "Slow travel & scenery", "Kyoto", "Quiet lanes, gardens, temples, and thoughtful pacing.", "Plan a 5-day slow-travel itinerary for Kyoto."),
    ("☼", "Coast & sunshine", "Amalfi Coast", "Sea views, small towns, and relaxed day trips.", "Plan a 5-day coastal escape along the Amalfi Coast."),
]
for column, (symbol, mood, place, detail, prompt_text) in zip(destination_cols, destinations):
    with column:
        st.markdown(
            f'<div class="destination-card"><div class="destination-symbol">{symbol}</div>'
            f'<div class="destination-mood">{mood}</div><h3>{place}</h3><p>{detail}</p></div>',
            unsafe_allow_html=True,
        )
        if st.button(f"Explore {place}", key=f"destination_{place}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()

st.markdown('<div class="section-kicker">Your personal travel desk</div><div class="section-title">Where shall we take you?</div>', unsafe_allow_html=True)

examples = [
    "Plan a 3-day food and culture trip to Paris.",
    "Find mapped hotels, attractions, and airports near Tokyo.",
    "What travel advisories should I check before visiting New York?",
    "Build a 5-day family itinerary for Istanbul on a moderate budget.",
]
example_cols = st.columns(4)
for column, prompt in zip(example_cols, examples):
    with column:
        if st.button(prompt, use_container_width=True, key=f"example_{examples.index(prompt)}"):
            st.session_state.pending_prompt = prompt
            st.rerun()

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Tell Roamly where you want to go, when, and what you enjoy...")
if not prompt:
    prompt = st.session_state.pop("pending_prompt", None)

if prompt:
    if not st.session_state.openai_key:
        st.warning("Add your OpenAI API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    conversation = []
    for item in st.session_state.messages:
        if item["role"] == "user":
            conversation.append(HumanMessage(content=item["content"]))
        else:
            conversation.append(AIMessage(content=item["content"]))

    with st.spinner("Roamly is researching and shaping your trip..."):
        try:
            answer, used_tools = run_travel_agent(
                conversation,
                st.session_state.openai_key,
                st.session_state.tavily_key,
            )
        except Exception as error:
            # Avoid echoing exception strings that could contain request metadata.
            st.error(f"The travel assistant could not complete this request ({type(error).__name__}). Check your API key and try again.")
            st.stop()

    if used_tools:
        answer += "\n\n---\n*Tools used: " + ", ".join(used_tools) + "*"
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

st.markdown(
    '<p class="note">Map data © <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap contributors</a>. Map listings may be incomplete; verify important details with providers. This app does not quote live airfare or confirm accommodation availability.</p>',
    unsafe_allow_html=True,
)
