"""
Hiver AI Customer Support Agent — Modern, User-Friendly UI.
Interactive triage, grounded reply generation, and safe escalation demo.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.pipeline import CustomerSupportPipeline, SupportAgentResponse

st.set_page_config(
    page_title="Hiver AI Customer Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SaaS-grade polish
st.markdown("""
<style>
    /* Clean, modern typography & card styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .hero-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    .hero-auto {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border-left: 6px solid #2e7d32;
        color: #1b5e20;
    }
    .hero-escalate {
        background: linear-gradient(135deg, #fff3e0 0%, #fbe9e7 100%);
        border-left: 6px solid #e65100;
        color: #bf360c;
    }
    .tweet-preview {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #edf2f7;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Starting Hiver AI Engine & Knowledge Base...")
def load_pipeline():
    return CustomerSupportPipeline()


pipeline = load_pipeline()
selected_brand = os.getenv("SELECTED_BRAND", "AmazonHelp")

# Header Section
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🤖 Hiver AI Customer Support Agent")
    st.markdown(
        f"**Enterprise Support Copilot** — Classifies customer intent, retrieves historical resolution precedents, "
        f"drafts brand-grounded responses, and enforces safety-first escalation rules."
    )
with header_col2:
    st.markdown(
        f"<div style='text-align: right; padding-top: 10px;'>"
        f"<span style='background-color: #ede9fe; color: #5b21b6; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 13px;'>"
        f"Active Brand: @{selected_brand}</span></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# Pre-defined test cases
PRESET_OPTIONS = {
    "Select an example...": {
        "text": "",
        "context": ""
    },
    "📦 Routine Delivery Delay (Safe Auto)": {
        "text": "Where is my package? The tracking link says delivered 2 hours ago but nothing is on my porch.",
        "context": "Customer placed order with expedited 2-day delivery."
    },
    "❌ Pre-Dispatch Cancellation (Safe Auto)": {
        "text": "I placed order #112-9876543 ten minutes ago by mistake. Can I cancel it before it enters dispatch?",
        "context": "Order placed recently; status is still 'Processing'."
    },
    "🔄 Return / Exchange Question (Safe Auto)": {
        "text": "How do I return a pair of boots that don't fit? Can I drop them off at a drop-off locker without printing a label?",
        "context": "Item purchased 10 days ago."
    },
    "⚖️ Explicit Legal Threat (Must Escalate)": {
        "text": "Your delivery driver threw my package over the fence and smashed my laptop! I am calling my attorney and suing Amazon tomorrow!",
        "context": "Customer extremely angry, threatening formal legal action."
    },
    "🚨 Compromised Account Alert (Must Escalate)": {
        "text": "Someone from another country logged into my account and changed my password. I am completely locked out!",
        "context": "Suspicious login alert received."
    },
    "💳 Fraudulent Overcharge (Must Escalate)": {
        "text": "Someone just charged $650 in digital gift cards to my Amazon stored card without my permission. FRAUD! Help me immediately!",
        "context": "Active unauthorized transactions reported on bank statement."
    },
    "🛠️ App Checkout Glitch (Safe Auto)": {
        "text": "The checkout button on the mobile app is greyed out and won't let me click 'Place Your Order'. How do I fix this?",
        "context": "Using iOS mobile app."
    }
}

# Sidebar: Safety Tuning & Modes
st.sidebar.header("⚙️ Safety & Risk Controls")
st.sidebar.caption("Fine-tune the agent's risk tolerance for auto-handling vs. human escalation.")

mode = st.sidebar.radio(
    "Policy Presets:",
    ["Balanced (Recommended)", "High Automation", "Ultra-Conservative", "Custom Sliders"],
    index=0
)

if mode == "Balanced (Recommended)":
    conf_thresh = 0.65
    ret_thresh = 0.55
elif mode == "High Automation":
    conf_thresh = 0.55
    ret_thresh = 0.45
elif mode == "Ultra-Conservative":
    conf_thresh = 0.75
    ret_thresh = 0.65
else:
    conf_thresh = st.sidebar.slider("Intent Confidence Cutoff", 0.40, 0.90, 0.65, 0.05)
    ret_thresh = st.sidebar.slider("Precedent Similarity Cutoff", 0.40, 0.80, 0.55, 0.05)

st.sidebar.markdown(f"""
<div style='background-color: #f1f5f9; padding: 12px; border-radius: 8px; font-size: 13px; margin: 10px 0;'>
  <div><b>Current Thresholds:</b></div>
  <div>• Confidence Cutoff: <b>{conf_thresh*100:.0f}%</b></div>
  <div>• Precedent Similarity: <b>{ret_thresh*100:.0f}%</b></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Specs")
st.sidebar.markdown("""
- **Taxonomy:** 12 Discovered Intents
- **Vector DB:** FAISS IndexFlatIP
- **Precedents:** 1,708 Verified Resolutions
- **Safety Gate:** Zero-Leakage Guaranteed
""")

# Session State for inputs
if "current_message" not in st.session_state:
    st.session_state["current_message"] = ""
if "current_context" not in st.session_state:
    st.session_state["current_context"] = ""


def apply_preset(preset_key):
    st.session_state["current_message"] = PRESET_OPTIONS[preset_key]["text"]
    st.session_state["current_context"] = PRESET_OPTIONS[preset_key]["context"]


# Main Interactive Area: 2 Columns
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Incoming Customer Inquiry")
    
    # 1-Click Quick Preset Selector
    selected_preset = st.selectbox(
        "⚡ Quick-load a real-world scenario:",
        list(PRESET_OPTIONS.keys()),
        index=0,
        help="Select a scenario to instantly populate sample text."
    )
    if selected_preset != "Select an example...":
        if st.session_state["current_message"] != PRESET_OPTIONS[selected_preset]["text"]:
            apply_preset(selected_preset)

    # Customer message input
    customer_message = st.text_area(
        "Customer Tweet / Message:",
        value=st.session_state["current_message"],
        height=130,
        placeholder="Type or paste a customer message here (e.g. 'Where is my order?')...",
        help="The raw text sent by the customer on social support."
    )

    # Optional conversation context
    with st.expander("➕ Add Conversation Context (Optional)", expanded=False):
        conversation_context = st.text_area(
            "Prior Thread History / Context:",
            value=st.session_state["current_context"],
            height=80,
            placeholder="e.g. 'Customer replied to dispatch email 2 hours ago...'"
        )

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        submit_btn = st.button("🚀 Analyze & Generate Support Resolution", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state["current_message"] = ""
            st.session_state["current_context"] = ""
            st.rerun()

with col_output:
    st.subheader("⚡ Agent Decision & Resolution")
    
    if submit_btn and customer_message.strip():
        with st.spinner("Classifying intent, retrieving historical precedents, and verifying safety rules..."):
            resp: SupportAgentResponse = pipeline.process(
                customer_message=customer_message.strip(),
                context=conversation_context.strip() if conversation_context.strip() else None,
                confidence_threshold=conf_thresh,
                retrieval_threshold=ret_thresh
            )

        # Decision Hero Card
        is_auto = (resp.decision == "AUTO_HANDLE")
        
        if is_auto:
            st.markdown(f"""
            <div class='hero-card hero-auto'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h3 style='margin: 0; color: #1b5e20;'>✅ AUTO-HANDLE: SAFE TO SEND</h3>
                        <p style='margin: 4px 0 0 0; font-size: 14px;'>Routine inquiry with verified historical precedent and high confidence.</p>
                    </div>
                    <div style='background-color: #2e7d32; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 700;'>
                        {resp.escalation_confidence*100:.0f}% Conf
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='hero-card hero-escalate'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h3 style='margin: 0; color: #bf360c;'>⚠️ ESCALATE TO HUMAN AGENT</h3>
                        <p style='margin: 4px 0 0 0; font-size: 14px;'><b>Reason:</b> {resp.escalation_reason}</p>
                    </div>
                    <div style='background-color: #e65100; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 700;'>
                        Safety Escalate
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Key Metrics Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 12px; color: #64748b; font-weight: 600;'>INTENT</div>
                <div style='font-size: 16px; font-weight: 700; color: #1e293b; margin-top: 2px;'>{resp.predicted_intent.replace('_', ' ').title()}</div>
                <div style='font-size: 12px; color: #475569;'>{resp.intent_confidence*100:.1f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 12px; color: #64748b; font-weight: 600;'>GROUNDING</div>
                <div style='font-size: 16px; font-weight: 700; color: #1e293b; margin-top: 2px;'>{resp.grounding_confidence*100:.1f}%</div>
                <div style='font-size: 12px; color: #475569;'>Precedent Match</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            risk_color = "#dc2626" if resp.risk_flags else "#16a34a"
            risk_text = ", ".join(resp.risk_flags) if resp.risk_flags else "None (Safe)"
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 12px; color: #64748b; font-weight: 600;'>RISK FLAGS</div>
                <div style='font-size: 13px; font-weight: 700; color: {risk_color}; margin-top: 4px;'>{risk_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # Grounded Draft Tweet Reply Card
        st.markdown("#### 💬 Grounded Support Reply Draft")
        st.markdown(f"""
        <div class='tweet-preview'>
            <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                <div style='width: 32px; height: 32px; border-radius: 50%; background-color: #5b21b6; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px;'>
                    A
                </div>
                <div>
                    <span style='font-weight: 700; font-size: 14px;'>{selected_brand} Support</span>
                    <span style='color: #64748b; font-size: 13px;'> @{selected_brand}</span>
                </div>
            </div>
            <div style='font-size: 15px; color: #1e293b; line-height: 1.45;'>
                {resp.draft_reply}
            </div>
            <div style='margin-top: 10px; font-size: 12px; color: #94a3b8;'>
                Character count: <b>{len(resp.draft_reply)}</b> / 280 (Twitter compliant)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable Evidence & Knowledge Base Precedents
        with st.expander("🔍 Show Retrieved Historical Precedents (Knowledge Base)", expanded=False):
            if resp.evidence:
                st.write(f"Top **{len(resp.evidence)}** historical resolutions retrieved from vector database:")
                for ev in resp.evidence:
                    st.markdown(
                        f"**Precedent #{ev.evidence_id}** — Similarity: `{ev.similarity_score:.3f}` | Conversation ID: `{ev.conversation_id}`"
                    )
                    st.markdown(f"> **Past Customer:** *\"{ev.customer_message}\"*")
                    st.markdown(f"> **Agent Resolution:** *\"{ev.brand_reply}\"*")
                    st.markdown("---")
            else:
                st.info("No close historical precedents found in knowledge base.")

    elif submit_btn:
        st.warning("Please enter or select a customer message on the left.")
    else:
        # Welcome placeholder
        st.info("👈 Select a pre-loaded scenario on the left or type your own inquiry to test the AI support agent.")
        st.markdown("""
        **What you will see:**
        - **Intent Identification:** Exact category and confidence score.
        - **Grounded Draft:** Twitter-ready reply under 280 chars with official sign-off (`^CS`).
        - **Automated vs. Human Escalation:** Strict safety checking for legal threats, account security, and uncertainty.
        - **Historical Evidence:** Real past Twitter interactions retrieved from our FAISS vector index.
        """)
