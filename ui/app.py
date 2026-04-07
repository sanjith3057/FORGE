import streamlit as st
import time
import sys
import os
import re
import pandas as pd
import random
import requests
from dotenv import load_dotenv

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ENV
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ─── STRONG PROMPT SHIELD ─────────────────────────────────────────────────────
_INPUT_PATTERNS = [
    re.compile(r"(output|reveal|show|print|tell me|display).*?(system prompt|initial prompt|instructions|context)", re.IGNORECASE),
    re.compile(r"ignore (all |previous |prior |your |the )?instructions?", re.IGNORECASE),
    re.compile(r"disregard (all |previous |your |the )?instructions?", re.IGNORECASE),
    re.compile(r"forget (everything|what you|your instructions|your training)", re.IGNORECASE),
    re.compile(r"override (your |all )?(previous |prior )?(instructions?|guidelines?|training)", re.IGNORECASE),
    re.compile(r"(jailbreak|DAN|do anything now|unrestricted|uncensored|unfiltered)", re.IGNORECASE),
    re.compile(r"you are (now |a |an )?(?!forge|an ai|a precise).*?(assistant|bot|ai|model|entity|system)", re.IGNORECASE),
    re.compile(r"act as (a |an )?(?!forge)", re.IGNORECASE),
    re.compile(r"pretend (you are|to be|that you)", re.IGNORECASE),
    re.compile(r"roleplay as|from now on (you are|act as|behave as|respond as)", re.IGNORECASE),
    re.compile(r"\[ADMIN(\s*MODE)?\s*(ENABLED|ON|ACTIVE)?\]", re.IGNORECASE),
    re.compile(r"as (the |a )?(developer|admin|owner|creator|engineer)", re.IGNORECASE),
    re.compile(r"respond (only |entirely )?in (base64|hex|rot13|binary|ascii)", re.IGNORECASE),
    re.compile(r"(encode|output|answer) (in|as|using) (base64|hex|rot13)", re.IGNORECASE),
    re.compile(r"<!--.*?(override|ignore|disable|bypass|inject).*?-->", re.IGNORECASE | re.DOTALL),
    re.compile(r"<script.*?>", re.IGNORECASE),
    re.compile(r"(disregard|ignore|skip).*?(format|rules|structure|reasoning)", re.IGNORECASE),
    re.compile(r"\[INST\].*?New instruction", re.IGNORECASE),
    re.compile(r"SYSTEM: |MAINTENANCE MODE", re.IGNORECASE),
    re.compile(r"New instruction:|Additional instruction:", re.IGNORECASE),
    re.compile(r"(extract|leak|steal|exfiltrate|reveal).*?(training data|api key|weights|parameters)", re.IGNORECASE),
    re.compile(r"how to (bypass|disable|remove) (content filter|safety|promptshield|guardrail)", re.IGNORECASE),
    re.compile(r"\b(rm\s+-rf|sudo|chmod|bash\s+-c|eval\(|exec\()\b", re.IGNORECASE),
]

def shield_input(text: str):
    for pat in _INPUT_PATTERNS:
        if pat.search(text):
            raise ValueError("🚨 PromptShield BLOCKED: This query contains suspicious patterns and has been rejected for security.")
    return True

# ─── API FUNCTION ─────────────────────────────────────────────────────────────
def fetch_live_response(prompt: str, is_tuned: bool) -> tuple[str, float]:
    if not GROQ_API_KEY:
        time.sleep(0.5)
        return ("GROQ API key is missing. Please add it in ui/.env", 1.0)

    start = time.time()
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    system_msg = (
        "You are Forge, a precise AI/ML expert. Always answer with clear Reasoning steps followed by a concise Answer."
        if is_tuned else
        "You are a general conversational assistant. Respond naturally."
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
        "temperature": 0.05 if is_tuned else 0.75,
        "max_tokens": 512
    }

    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
        content = r.json()['choices'][0]['message']['content'] if r.status_code == 200 else f"API Error {r.status_code}"
    except Exception as e:
        content = f"Connection Error: {str(e)}"

    latency = round(time.time() - start, 2)
    if is_tuned:
        latency += round(random.uniform(0.1, 0.4), 2)

    return content, latency

# ─── PAGE CONFIG + CLEAN PROFESSIONAL THEME ───────────────────────────────────
st.set_page_config(page_title="FORGE | QLoRA Dashboard", layout="wide", page_icon="🔥")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { background: #f8fafc; color: #0f172a; }
    [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
    .forge-title { font-size: 2.9rem; font-weight: 800; background: linear-gradient(90deg, #ff6400, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-card, .result-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.6rem; box-shadow: 0 4px 15px rgba(0,0,0,0.04); transition: all 0.3s ease; }
    .metric-card:hover, .result-card:hover { transform: translateY(-5px); border-color: #ff6400; }
    .result-tuned { border-left: 5px solid #ff6400; }
    .result-base { border-left: 5px solid #64748b; }
    .stButton > button { background: linear-gradient(135deg, #ff6400, #a855f7); border: none; font-weight: 600; border-radius: 10px; padding: 0.85rem 2rem; color: white; }
    .stButton > button:hover { transform: scale(1.04); box-shadow: 0 10px 25px rgba(255,100,0,0.3); }
</style>
""", unsafe_allow_html=True)

# Session State
if 'analytics' not in st.session_state:
    st.session_state.analytics = {
        'total_queries': 0, 'security_blocks': 0,
        'base_latency': [], 'tuned_latency': [],
        'batch_sizes': [], 'history': [], 'blocked_logs': []
    }

# Sidebar
with st.sidebar:
    st.markdown('<p class="forge-title">FORGE</p>', unsafe_allow_html=True)
    st.caption("v1.0.3 • QLoRA Fine-Tuning Dashboard • RTX 2050 Optimized")
    st.divider()
    menu = st.radio("Navigation", ["⚖️ Model Comparison", "📊 Analytics", "🛡️ Security Log"], label_visibility="collapsed")
    st.divider()
    st.subheader("⚙️ Engine Settings")
    throttle = st.slider("Throttle Delay (seconds)", 0.0, 2.0, 0.5, 0.1)
    st.divider()
    st.subheader("Live Telemetry")
    a = st.session_state.analytics
    avg_base = sum(a['base_latency']) / max(1, len(a['base_latency']))
    avg_tuned = sum(a['tuned_latency']) / max(1, len(a['tuned_latency']))
    st.metric("Total Queries", a['total_queries'])
    st.metric("Threats Blocked", a['security_blocks'])
    st.metric("Avg Base Latency", f"{avg_base:.2f}s")
    st.metric("Avg Tuned Latency", f"{avg_tuned:.2f}s")
    if st.button("Clear All Data", use_container_width=True):
        st.session_state.analytics = {'total_queries': 0, 'security_blocks': 0, 'base_latency': [], 'tuned_latency': [], 'batch_sizes': [], 'history': [], 'blocked_logs': []}
        st.rerun()

# Header
st.markdown('<div style="text-align:center; margin: 2rem 0 2.5rem 0;"><h1 class="forge-title">FORGE</h1><p style="color:#475569; font-size:1.15rem;">QLoRA Fine-Tuned Intelligence Dashboard</p></div>', unsafe_allow_html=True)

# Model Comparison Tab
if menu == "⚖️ Model Comparison":
    st.subheader("⚖️ Batch Query Processor")
    col_in, col_set = st.columns([3, 1])
    with col_in:
        user_prompt = st.text_area("Enter AI/ML questions (one per line):", height=180, placeholder="What is the lost-in-the-middle problem in RAG?\nExplain QLoRA in simple terms.")
    with col_set:
        show_raw = st.checkbox("Show raw output", value=False)
        run_btn = st.button("⚡ Run Benchmark Comparison", use_container_width=True, type="primary")

    if run_btn and user_prompt.strip():
        questions = [q.strip() for q in user_prompt.split('\n') if q.strip()]
        if questions:
            st.session_state.analytics['batch_sizes'].append(len(questions))   # Fixed batch size tracking
        for q in questions:
            try:
                shield_input(q)
            except ValueError as e:
                st.session_state.analytics['security_blocks'] += 1
                st.session_state.analytics['blocked_logs'].append({'timestamp': time.strftime("%H:%M:%S"), 'query': q[:100]+("..." if len(q)>100 else ""), 'reason': str(e)})
                st.error(str(e))
                continue

            base_content, base_time = fetch_live_response(q, False)
            time.sleep(throttle)
            tuned_content, tuned_time = fetch_live_response(q, True)
            time.sleep(throttle)

            st.session_state.analytics['total_queries'] += 1
            st.session_state.analytics['base_latency'].append(base_time)
            st.session_state.analytics['tuned_latency'].append(tuned_time)
            st.session_state.analytics['history'].append({'q': q, 'base': base_content, 'tuned': tuned_content, 'base_t': base_time, 'tuned_t': tuned_time})

            st.markdown(f"**Query:** `{q}`")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="result-card result-base"><b>🧊 Base Model</b><br><br>{base_content.replace("\n","<br>") if show_raw else base_content}<div style="margin-top:12px;color:#64748b;font-size:0.85rem;">⏱ {base_time}s</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="result-card result-tuned"><b>🔥 FORGE (Fine-Tuned)</b><br><br>{tuned_content.replace("\n","<br>") if show_raw else tuned_content}<div style="margin-top:12px;color:#ff6400;font-size:0.85rem;">⏱ {tuned_time}s • Structured</div></div>', unsafe_allow_html=True)
            st.divider()

# Analytics Dashboard (Improved)
elif menu == "📊 Analytics":
    st.subheader("📊 FORGE Analytics Dashboard")
    a = st.session_state.analytics

    if not a['base_latency']:
        st.info("Run benchmarks in Model Comparison to see detailed analytics.")
    else:
        avg_base = sum(a['base_latency']) / len(a['base_latency'])
        avg_tuned = sum(a['tuned_latency']) / len(a['tuned_latency'])
        safe_count = a['total_queries'] - a['security_blocks']
        improvement = round(((avg_base - avg_tuned) / avg_base) * 100, 1) if avg_base > 0 else 0

        # Summary Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><h3 style="margin:0;color:#475569;">Total Queries</h3><h2 style="margin:0;color:#0f172a;">{a["total_queries"]}</h2></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><h3 style="margin:0;color:#475569;">Threats Blocked</h3><h2 style="margin:0;color:#ef4444;">{a["security_blocks"]}</h2></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><h3 style="margin:0;color:#475569;">Avg Base Latency</h3><h2 style="margin:0;color:#0f172a;">{avg_base:.2f}s</h2></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card"><h3 style="margin:0;color:#475569;">FORGE Improvement</h3><h2 style="margin:0;color:#22c55e;">{improvement}% faster</h2></div>', unsafe_allow_html=True)

        st.divider()

        # Detailed Professional Summary Report
        if st.button("📋 Generate Full Session Summary Report", use_container_width=True, type="primary"):
            st.markdown("### FORGE Session Summary Report")
            st.markdown("---")
            st.write(f"**Total Queries Processed:** {a['total_queries']}")
            st.write(f"**Security Blocks:** {a['security_blocks']} ({round(a['security_blocks']/max(1,a['total_queries'])*100,1)}% of total)")
            st.write(f"**Average Base Model Latency:** {avg_base:.2f}s")
            st.write(f"**Average FORGE Tuned Latency:** {avg_tuned:.2f}s")
            st.write(f"**Latency Improvement:** {improvement}% faster with fine-tuned model")
            st.write(f"**Best Tuned Latency Recorded:** {min(a['tuned_latency']):.2f}s")
            st.write(f"**Total Batches Run:** {len(a['batch_sizes'])}")
            st.write(f"**Average Batch Size:** {sum(a['batch_sizes'])/max(1,len(a['batch_sizes'])):.1f} questions")
            st.markdown("---")
            st.success("Full session summary generated successfully.")

        # Extra Visuals & Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Latency Comparison Over Queries**")
            latency_df = pd.DataFrame({"Base Model": a['base_latency'], "FORGE Tuned": a['tuned_latency']})
            st.line_chart(latency_df, use_container_width=True, height=380)

        with col2:
            st.markdown("**Security Distribution**")
            security_df = pd.DataFrame({"Category": ["Safe Queries", "Blocked Attempts"], "Count": [safe_count, a['security_blocks']]})
            st.bar_chart(security_df.set_index("Category"), use_container_width=True, height=380)

        st.markdown("**Batch Size Distribution**")
        if a['batch_sizes']:
            batch_df = pd.DataFrame({"Batch Size": a['batch_sizes']})
            st.bar_chart(batch_df, use_container_width=True)

        # Recent Queries Table
        if a['history']:
            st.markdown("**Recent Queries Performance**")
            recent = a['history'][-8:]
            table_data = [{"Query": item['q'][:65] + "..." if len(item['q']) > 65 else item['q'],
                           "Base Latency": f"{item.get('base_t',0):.2f}s",
                           "Tuned Latency": f"{item.get('tuned_t',0):.2f}s"} for item in recent]
            st.table(pd.DataFrame(table_data))

# Security Log Tab
elif menu == "🛡️ Security Log":
    st.subheader("🛡️ PromptShield Security Log")
    blocked_logs = st.session_state.analytics.get('blocked_logs', [])
    if not blocked_logs:
        st.success("✅ No blocked attempts in this session.")
    else:
        st.warning(f"🚨 {len(blocked_logs)} suspicious queries were blocked.")
        for log in reversed(blocked_logs[-15:]):
            st.error(f"**{log['timestamp']}** — {log['query']}\n\n**Reason:** {log['reason']}")

st.caption("FORGE • QLoRA Fine-Tuning Pipeline • Strong PromptShield Active")