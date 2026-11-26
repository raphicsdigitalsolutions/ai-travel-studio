"""Roamly — immersive AI Travel Studio."""

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

# ----------------------------- Visual system -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
      --ink:#103b3b; --ink-2:#1b5651; --muted:#6a8984; --teal:#078b78;
      --mint:#dff8ec; --mint-2:#c8f0df; --gold:#c5a46a;
      --glass:rgba(255,255,255,.72); --line:rgba(255,255,255,.88);
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp {
      color:var(--ink);
      background:
        radial-gradient(ellipse at 8% 5%,rgba(48,205,157,.18),transparent 34rem),
        radial-gradient(ellipse at 95% 25%,rgba(33,153,133,.15),transparent 32rem),
        radial-gradient(ellipse at 55% 100%,rgba(202,236,218,.35),transparent 36rem),
        linear-gradient(145deg,#f6fcf8 0%,#eaf8f1 48%,#f5fbf7 100%);
    }
    [data-testid="stHeader"] { background:rgba(245,252,248,.72); }
    [data-testid="stSidebar"] {
      background:linear-gradient(180deg,#e0f5eb 0%,#effaf4 55%,#e5f5ec 100%);
      border-right:1px solid rgba(17,122,101,.12);
    }
    [data-testid="stSidebar"] > div { padding-top:1.35rem; }
    .block-container { max-width:1440px; padding-top:1.7rem; padding-bottom:4rem; }
    .sidebar-brand { color:#14584f;font-family:'Manrope',sans-serif;font-weight:900;font-size:25px;letter-spacing:-.06em; }
    .sidebar-sub { margin-top:-3px;color:#688780;font-size:10px;letter-spacing:.19em;text-transform:uppercase; }
    .sidebar-eyebrow { color:#258b78;font-size:10px;font-weight:900;letter-spacing:.17em;text-transform:uppercase; }
    .sidebar-note { color:#718c8b;font-size:11px;line-height:1.7; }
    .topline { display:flex;align-items:center;gap:10px;color:#258b78;font-size:10px;font-weight:900;letter-spacing:.2em;text-transform:uppercase;margin:0 0 13px; }
    .live-dot { width:8px;height:8px;border-radius:50%;background:#20b68e;box-shadow:0 0 0 5px rgba(32,182,142,.12),0 0 18px rgba(32,182,142,.7); }
    .hero {
      position:relative;display:flex;align-items:center;justify-content:space-between;gap:25px;
      min-height:420px;padding:54px 58px;overflow:hidden;border-radius:34px;
      border:1px solid rgba(255,255,255,.94);
      background:
        radial-gradient(circle at 77% 48%,rgba(64,215,167,.28),transparent 20rem),
        radial-gradient(circle at 8% 110%,rgba(190,239,214,.55),transparent 24rem),
        linear-gradient(125deg,#fdfffd 0%,#e9f9f1 52%,#d6f2e5 100%);
      box-shadow:0 28px 75px rgba(20,100,81,.12),inset 0 1px 0 #fff;
      isolation:isolate;
    }
    .hero:before,.hero:after { content:"";position:absolute;border:1px solid rgba(13,146,117,.12);border-radius:50%;pointer-events:none;z-index:-1; }
    .hero:before { width:550px;height:550px;right:1%;top:-75px;box-shadow:0 0 0 28px rgba(13,146,117,.025),0 0 0 60px rgba(13,146,117,.018),0 0 0 95px rgba(13,146,117,.015); }
    .hero:after { width:400px;height:400px;right:6%;top:0; }
    .hero-copy { position:relative;z-index:4;max-width:620px; }
    .hero-kicker { display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid rgba(8,139,115,.16);border-radius:99px;background:rgba(255,255,255,.66);color:#278875;font-size:10px;font-weight:900;letter-spacing:.1em;text-transform:uppercase;margin-bottom:20px;box-shadow:0 8px 22px rgba(14,105,83,.05); }
    .hero-copy h1 { color:#103b3b;font-family:'Manrope',sans-serif;font-size:clamp(2.7rem,5vw,5.1rem);font-weight:800;line-height:.99;letter-spacing:-.075em;margin:0 0 19px; }
    .hero-copy h1 span { color:#0b9c80; }
    .hero-copy p { max-width:570px;color:#5c7e79;font-size:15px;line-height:1.8;margin:0; }
    .hero-tags { display:flex;flex-wrap:wrap;gap:9px;margin-top:24px; }
    .hero-tags span { padding:9px 12px;border:1px solid rgba(17,139,111,.14);border-radius:99px;background:rgba(255,255,255,.62);color:#286d63;font-size:10px;font-weight:800;box-shadow:0 8px 20px rgba(14,105,83,.04); }
    .hero-art { position:relative;z-index:2;display:grid;place-items:center;width:400px;height:330px;min-width:330px;perspective:1100px; }
    .planet-glow { position:absolute;width:285px;height:285px;border-radius:50%;background:rgba(35,202,152,.12);filter:blur(28px); }
    .planet {
      position:relative;width:222px;height:222px;border-radius:50%;
      background:
        radial-gradient(ellipse at 28% 22%,rgba(255,255,255,.95),transparent 16%),
        radial-gradient(ellipse at 37% 38%,#88f1c7 0%,#38d7a6 27%,#0ba98e 54%,#08766f 82%,#064d55 100%);
      transform:rotateX(12deg) rotateY(-19deg);
      box-shadow:inset -28px -26px 48px rgba(1,45,48,.42),inset 12px 10px 25px rgba(255,255,255,.54),0 34px 58px rgba(4,93,78,.27),0 0 70px rgba(27,199,151,.24);
      animation:planet-float 7s ease-in-out infinite alternate;overflow:hidden;
    }
    .planet:before {
      content:"";position:absolute;inset:8px;border-radius:50%;border:1px solid rgba(231,255,243,.31);
      background:
        radial-gradient(ellipse at 28% 32%,rgba(226,255,236,.78) 0 6%,transparent 6.7%),
        radial-gradient(ellipse at 58% 24%,rgba(226,255,236,.58) 0 3%,transparent 3.8%),
        radial-gradient(ellipse at 69% 53%,rgba(226,255,236,.55) 0 10%,transparent 11%),
        radial-gradient(ellipse at 40% 72%,rgba(226,255,236,.45) 0 8%,transparent 9%),
        radial-gradient(ellipse at 20% 62%,rgba(226,255,236,.35) 0 4%,transparent 5%);
      box-shadow:inset 0 0 32px rgba(255,255,255,.16);
    }
    .planet:after { content:"";position:absolute;inset:0;border-radius:50%;background:linear-gradient(115deg,transparent 20%,rgba(255,255,255,.34) 43%,transparent 58%);transform:translateX(-115%);animation:planet-shine 6s ease-in-out infinite; }
    .orbit { position:absolute;width:355px;height:142px;border:1.5px solid rgba(8,132,119,.42);border-radius:50%;transform:rotate(-24deg) rotateX(67deg);box-shadow:0 0 20px rgba(14,165,135,.12);animation:orbit-spin 22s linear infinite; }
    .orbit.two { width:315px;height:185px;border-color:rgba(8,132,119,.26);transform:rotate(58deg) rotateX(68deg);animation-duration:30s;animation-direction:reverse; }
    .orbit.three { width:280px;height:235px;border-color:rgba(193,162,100,.32);transform:rotate(20deg) rotateX(72deg);animation-duration:36s; }
    .orbit:after { content:"";position:absolute;top:9px;left:18%;width:11px;height:11px;border-radius:50%;background:#fff;box-shadow:0 0 0 5px rgba(22,177,139,.13),0 0 18px #27c99b; }
    .float-card { position:absolute;z-index:5;padding:11px 14px;border:1px solid rgba(255,255,255,.9);border-radius:15px;background:rgba(255,255,255,.78);backdrop-filter:blur(14px);box-shadow:0 14px 30px rgba(9,83,72,.13);color:#1a5e58;font-size:10px;font-weight:900; }
    .float-a { top:29px;left:0;transform:rotate(-5deg);animation:float-a 6s ease-in-out infinite alternate; }
    .float-b { bottom:30px;right:0;transform:rotate(4deg);animation:float-b 7s ease-in-out infinite alternate; }
    .float-c { top:52%;left:-16px;transform:rotate(-3deg);font-size:9px;color:#9a7945; }
    @keyframes planet-float { from{transform:rotateX(12deg) rotateY(-19deg) translateY(2px)} to{transform:rotateX(-7deg) rotateY(13deg) translateY(-12px)} }
    @keyframes planet-shine { 0%,32%{transform:translateX(-115%)} 75%,100%{transform:translateX(115%)} }
    @keyframes orbit-spin { from{rotate:0deg} to{rotate:360deg} }
    @keyframes float-a { from{translate:0 2px} to{translate:0 -9px} }
    @keyframes float-b { from{translate:0 -2px} to{translate:0 9px} }
    .section-head { display:flex;justify-content:space-between;align-items:end;gap:18px;margin:34px 0 15px; }
    .section-kicker { color:#258b78;font-size:10px;font-weight:900;letter-spacing:.18em;text-transform:uppercase;margin-bottom:7px; }
    .section-title { color:#153e40;font-family:'Manrope',sans-serif;font-size:25px;font-weight:800;letter-spacing:-.05em;margin:0; }
    .section-note { color:#78918c;font-size:11px; }
    .stat-row { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-top:17px; }
    .stat-card { display:flex;align-items:center;gap:12px;padding:15px;border:1px solid rgba(255,255,255,.92);border-radius:18px;background:linear-gradient(145deg,rgba(255,255,255,.78),rgba(226,248,237,.74));box-shadow:0 12px 25px rgba(18,100,83,.06),inset 0 1px 0 #fff; }
    .stat-icon { width:39px;height:39px;display:grid;place-items:center;border-radius:13px;background:linear-gradient(145deg,#d5f6e7,#fff);box-shadow:inset 0 1px 0 #fff,0 7px 12px rgba(8,139,115,.1);font-size:17px; }
    .stat-card strong { display:block;color:#1c514b;font-size:12px;font-weight:900; }
    .stat-card small { color:#78918c;font-size:10px; }
    .destination-card { position:relative;min-height:190px;padding:20px;border:1px solid rgba(255,255,255,.96);border-radius:23px;overflow:hidden;background:radial-gradient(circle at 100% 0,rgba(48,203,158,.22),transparent 9rem),linear-gradient(145deg,rgba(255,255,255,.94),rgba(225,246,236,.88));box-shadow:0 15px 32px rgba(18,100,83,.075),inset 0 1px 0 #fff;transition:transform .22s ease,box-shadow .22s ease; }
    .destination-card:after { content:"";position:absolute;width:120px;height:120px;right:-38px;bottom:-58px;border-radius:50%;border:1px solid rgba(11,156,128,.13);box-shadow:0 0 0 12px rgba(11,156,128,.025),0 0 0 26px rgba(11,156,128,.018); }
    .destination-symbol { width:38px;height:38px;display:grid;place-items:center;border-radius:13px;color:#07836d;background:linear-gradient(145deg,#c8f0df,#fff);box-shadow:inset 0 1px 0 #fff,0 8px 15px rgba(8,139,115,.11);font-size:19px;font-weight:900; }
    .destination-mood { margin-top:15px;color:#29927d;font-size:9px;font-weight:900;letter-spacing:.13em;text-transform:uppercase; }
    .destination-card h3 { position:relative;z-index:1;margin:4px 0 6px;color:#174548;font-family:'Manrope',sans-serif;font-size:20px;font-weight:800;letter-spacing:-.04em; }
    .destination-card p { position:relative;z-index:1;margin:0;color:#688382;font-size:11px;line-height:1.6;max-width:245px; }
    .feature-card { min-height:150px;padding:21px;border:1px solid rgba(255,255,255,.95);border-radius:21px;background:linear-gradient(145deg,rgba(255,255,255,.89),rgba(230,248,239,.87));box-shadow:0 13px 28px rgba(18,100,83,.065),inset 0 1px 0 #fff; }
    .feature-num { color:#28a98a;font-size:10px;font-weight:900;letter-spacing:.13em; }
    .feature-card h3 { margin:11px 0 7px;color:#19484a;font-size:14px;font-weight:900; }
    .feature-card p { margin:0;color:#668284;font-size:11px;line-height:1.65; }
    .stButton > button { min-height:42px;border:1px solid rgba(14,151,121,.19);border-radius:13px;background:linear-gradient(180deg,#22c99c,#07836d);color:white;font-weight:800;box-shadow:0 4px 0 #086b5a,0 11px 22px rgba(10,144,115,.13);transition:transform .18s ease,box-shadow .18s ease; }
    .stButton > button:hover { color:white;border-color:#0a9278;transform:translateY(-2px);box-shadow:0 6px 0 #086b5a,0 15px 25px rgba(10,144,115,.18); }
    [data-testid="stChatMessage"] { border:1px solid rgba(255,255,255,.85);border-radius:18px;background:rgba(255,255,255,.58); }
    [data-testid="stChatInput"] { border-radius:16px; }
    .chat-panel { padding:20px 22px;border:1px solid rgba(255,255,255,.92);border-radius:22px;background:rgba(255,255,255,.62);box-shadow:0 15px 30px rgba(18,100,83,.055); }
    .note { color:#718c8b;font-size:10px;line-height:1.7; }
    .gate-wrap { max-width:990px;margin:5vh auto 0;display:grid;grid-template-columns:1.06fr .94fr;min-height:570px;overflow:hidden;border-radius:34px;border:1px solid rgba(255,255,255,.96);background:linear-gradient(135deg,rgba(255,255,255,.92),rgba(220,246,233,.88));box-shadow:0 35px 100px rgba(14,93,75,.16),inset 0 1px 0 #fff; }
    .gate-art { position:relative;display:flex;flex-direction:column;justify-content:space-between;padding:42px;overflow:hidden;background:radial-gradient(circle at 50% 46%,rgba(61,217,165,.28),transparent 18rem),linear-gradient(150deg,#e4f8ed,#cfeee0); }
    .gate-art:before,.gate-art:after { content:"";position:absolute;border:1px solid rgba(8,132,119,.13);border-radius:50%;pointer-events:none; }
    .gate-art:before { width:420px;height:420px;left:3%;top:18%;box-shadow:0 0 0 25px rgba(8,132,119,.025),0 0 0 50px rgba(8,132,119,.02); }
    .gate-brand { position:relative;z-index:3;font-family:'Manrope',sans-serif;font-size:23px;font-weight:900;color:#14584f;letter-spacing:-.06em; }
    .gate-caption { position:relative;z-index:3;max-width:300px;color:#51756e;font-size:13px;line-height:1.75; }
    .gate-planet-wrap { position:absolute;inset:0;display:grid;place-items:center;perspective:900px; }
    .gate-planet { position:relative;width:190px;height:190px;border-radius:50%;background:radial-gradient(circle at 28% 24%,#fff,transparent 17%),radial-gradient(circle at 35% 35%,#78e9bb,#1bbd91 46%,#076a66 85%);box-shadow:inset -22px -22px 40px rgba(1,50,52,.32),inset 9px 8px 22px rgba(255,255,255,.5),0 35px 60px rgba(4,93,78,.25),0 0 70px rgba(27,199,151,.25);transform:rotateX(14deg) rotateY(-20deg);animation:planet-float 7s ease-in-out infinite alternate; }
    .gate-ring { position:absolute;width:300px;height:120px;border:1px solid rgba(9,128,111,.42);border-radius:50%;transform:rotate(-26deg) rotateX(66deg);animation:orbit-spin 24s linear infinite; }
    .gate-form { display:flex;flex-direction:column;justify-content:center;padding:48px 44px; }
    .gate-kicker { color:#27947d;font-size:10px;font-weight:900;letter-spacing:.19em;text-transform:uppercase; }
    .gate-form h1 { color:#153e40;font-family:'Manrope',sans-serif;font-size:31px;font-weight:900;line-height:1.08;letter-spacing:-.06em;margin:12px 0 10px; }
    .gate-form p { color:#688382;font-size:12px;line-height:1.7;margin:0 0 22px; }
    .gate-foot { color:#879b96;font-size:10px;line-height:1.6;margin-top:14px; }
    .stForm { border:0;padding:0;background:transparent; }
    @media(max-width:900px) { .hero { padding:38px 30px;min-height:0; } .hero-art { width:320px;min-width:250px; } .planet { width:175px;height:175px; } .stat-row { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media(max-width:700px) { .block-container { padding-top:1rem; } .hero { flex-direction:column;align-items:flex-start;padding:30px 24px; } .hero-art { width:100%;height:265px;min-width:0; } .hero-copy h1 { font-size:clamp(2.8rem,13vw,4rem); } .stat-row { grid-template-columns:repeat(2,minmax(0,1fr)); } .gate-wrap { grid-template-columns:1fr;margin:1vh auto; } .gate-art { min-height:230px;padding:25px; } .gate-art:before { width:260px;height:260px;left:20%;top:-5%; } .gate-planet { width:120px;height:120px; } .gate-ring { width:210px;height:88px; } .gate-caption { max-width:240px;font-size:11px; } .gate-form { padding:30px 24px; } .gate-form h1 { font-size:27px; } }
    @media(prefers-reduced-motion:reduce) { *,*:before,*:after { animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _secret(name: str, fallback: str = "") -> str:
    """Read a deployment secret without displaying it."""
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
if "show_connection_gate" not in st.session_state:
    st.session_state.show_connection_gate = not bool(st.session_state.openai_key)


def render_connection_gate() -> None:
    """Blocking first-run API setup; avoids dismissible modal/session-state quirks."""
    st.markdown(
        """
        <div class="gate-wrap">
          <section class="gate-art">
            <div class="gate-brand">🌍 Roamly</div>
            <div class="gate-planet-wrap"><div class="gate-ring"></div><div class="gate-planet"></div></div>
            <div class="gate-caption">A calmer way to discover the world. Build thoughtful journeys with an AI travel companion.</div>
          </section>
          <section class="gate-form">
            <div class="gate-kicker">Start your journey</div>
            <h1>Connect your travel intelligence.</h1>
            <p>Enter an OpenAI API key to unlock personalised trip planning. Add Tavily optionally for current destination research and travel advisories.</p>
          </section>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Streamlit widgets cannot be inserted inside the HTML card, so use a
    # centered form directly below the visual card with matching visual style.
    left, center, right = st.columns([1, 2, 1])
    with center:
        with st.form("blocking_api_connection_form", clear_on_submit=False):
            openai_input = st.text_input(
                "OpenAI API key",
                type="password",
                placeholder="sk-...",
                help="Required. Your key stays in this Streamlit session and is not written to GitHub.",
            )
            tavily_input = st.text_input(
                "Tavily API key (optional)",
                type="password",
                placeholder="tvly-...",
                help="Optional: enables current web research.",
            )
            trust = st.checkbox("I trust this deployment with the API keys I enter.")
            submitted = st.form_submit_button(
                "Connect securely and enter Roamly",
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
                    st.session_state.show_connection_gate = False
                    st.rerun()
        st.markdown(
            '<div class="gate-foot">Your key is used only to make requests to OpenAI for this session. Never enter API keys on a deployment you do not control or trust.</div>',
            unsafe_allow_html=True,
        )


# This gate is deliberately blocking when no key is available. It is not a
# dismissible st.dialog, so first-run onboarding cannot silently disappear.
if st.session_state.show_connection_gate or not st.session_state.openai_key:
    render_connection_gate()
    st.stop()


# -------------------------------- Sidebar ---------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌍 Roamly</div><div class="sidebar-sub">AI Travel Studio</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-eyebrow">Your connection</div>', unsafe_allow_html=True)
    st.success("OpenAI connected")
    if st.session_state.tavily_key:
        st.caption("Live web research enabled")
    else:
        st.caption("OpenStreetMap discovery enabled · Tavily optional")
    if st.button("Connect / update API keys", use_container_width=True):
        st.session_state.show_connection_gate = True
        st.rerun()
    if st.button("New trip", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("#### Your travel toolkit")
    st.markdown("- AI trip itineraries\n- Mapped hotels and restaurants\n- Airports and local attractions\n- Flight-search links\n- Optional live travel research")
    st.markdown(
        '<p class="sidebar-note">Map listings can be incomplete. Confirm schedules, prices and availability with providers before booking.</p>',
        unsafe_allow_html=True,
    )


# --------------------------------- Hero -----------------------------------
st.markdown(
    """
    <div class="topline"><span class="live-dot"></span> Your next trip starts here</div>
    <section class="hero">
      <div class="hero-copy">
        <div class="hero-kicker">✦ Your intelligent travel companion</div>
        <h1>Travel more<br><span>thoughtfully.</span></h1>
        <p>Discover destinations, compare ideas, and shape a practical itinerary around your dates, budget, and travel style. One conversation can take you from inspiration to a plan.</p>
        <div class="hero-tags">
          <span>✦ AI trip planner</span><span>⌖ Open map discovery</span><span>↗ Live research</span>
        </div>
      </div>
      <div class="hero-art" aria-hidden="true">
        <div class="planet-glow"></div>
        <div class="orbit"></div><div class="orbit two"></div><div class="orbit three"></div>
        <div class="planet"></div>
        <div class="float-card float-a">⌖ Discover places</div>
        <div class="float-card float-b">✦ Your itinerary</div>
        <div class="float-card float-c">✧ Explore smarter</div>
      </div>
    </section>
    <div class="stat-row">
      <div class="stat-card"><div class="stat-icon">✦</div><div><strong>Personal plans</strong><small>Built around your trip</small></div></div>
      <div class="stat-card"><div class="stat-icon">⌖</div><div><strong>Place discovery</strong><small>OpenStreetMap data</small></div></div>
      <div class="stat-card"><div class="stat-icon">↗</div><div><strong>Current context</strong><small>Optional live research</small></div></div>
      <div class="stat-card"><div class="stat-icon">◈</div><div><strong>Travel-ready</strong><small>Useful links and ideas</small></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------- Destination discovery ------------------------
st.markdown(
    '<div class="section-head"><div><div class="section-kicker">Explore your next chapter</div><h2 class="section-title">Where could your story go?</h2></div><div class="section-note">Choose a direction to start planning</div></div>',
    unsafe_allow_html=True,
)
destination_cols = st.columns(3, gap="medium")
destinations = [
    ("✦", "Culture & city lights", "Istanbul", "Historic streets, waterfront views, and unforgettable local food.", "Plan a 4-day culture and food trip to Istanbul."),
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


# ------------------------------- Capabilities -----------------------------
st.markdown(
    '<div class="section-head"><div><div class="section-kicker">Plan beyond the ordinary</div><h2 class="section-title">A complete travel desk</h2></div></div>',
    unsafe_allow_html=True,
)
feature_cols = st.columns(3, gap="medium")
features = [
    ("01", "Discover destinations", "Find mapped hotels, restaurants, attractions, museums and airports."),
    ("02", "Get current context", "Research destination information, travel advisories and recent updates."),
    ("03", "Shape your itinerary", "Turn your dates, interests and budget into a practical trip plan."),
]
for column, (number, title, description) in zip(feature_cols, features):
    with column:
        st.markdown(
            f'<div class="feature-card"><div class="feature-num">{number}</div><h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )


# --------------------------------- Planner -------------------------------
st.markdown(
    '<div class="section-head"><div><div class="section-kicker">Your personal travel desk</div><h2 class="section-title">Let’s plan something memorable.</h2></div></div>',
    unsafe_allow_html=True,
)
quick_prompts = [
    "Plan a 3-day food and culture trip to Paris.",
    "Find mapped hotels and attractions near Tokyo.",
    "What should I know before visiting New York?",
    "Build a 5-day family itinerary for Istanbul.",
]
prompt_cols = st.columns(4, gap="small")
for index, (column, quick_prompt) in enumerate(zip(prompt_cols, quick_prompts)):
    with column:
        if st.button(quick_prompt, key=f"quick_{index}", use_container_width=True):
            st.session_state.pending_prompt = quick_prompt
            st.rerun()

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Where do you want to go, when, and what do you enjoy?")
if not prompt:
    prompt = st.session_state.pop("pending_prompt", None)

if prompt:
    if not st.session_state.openai_key:
        st.session_state.show_connection_gate = True
        st.rerun()

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
            st.error(
                f"The travel assistant could not complete this request ({type(error).__name__}). "
                "Check your API key and try again."
            )
            st.stop()

    if used_tools:
        answer += "\n\n---\n*Tools used: " + ", ".join(used_tools) + "*"
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


st.markdown(
    '<p class="note">Map data © <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap contributors</a>. Map listings may be incomplete; verify important details with providers. This app does not quote live airfare or confirm accommodation availability.</p>',
    unsafe_allow_html=True,
)
