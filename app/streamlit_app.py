"""
Streamlit Demonstration UI for AI Customer Support Agent.
Allows evaluators to enter customer messages and conversation context,
and view:
- Predicted Intent & Confidence
- AUTO-HANDLE vs ESCALATE routing decision & reason
- Grounded Draft Reply
- Retrieved Historical Precedents (Evidence drawer)
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
    page_title="AI Customer Support Agent | AmazonHelp",
    page_icon="📦",
    layout="wide"
)


@st.cache_resource(show_spinner="Loading models & FAISS index...")
def get_pipeline():
    return CustomerSupportPipeline()


pipeline = get_pipeline()

st.title("📦 Amazon AI Customer Support Agent")
st.caption("Demonstration prototype grounded in historical Amazon Twitter customer support resolutions.")

# Sidebar Quick Test Presets
st.sidebar.header("🧪 Pre-loaded Test Cases")
PRESETS = {
    "Select a preset...": {
        "message": "",
        "context": ""
    },
    "1. Routine Tracking Delay (Safe Auto)": {
        "message": "Where is my package? The tracking link says delivered 2 hours ago but nothing is on my porch.",
        "context": "Ordered 3 days ago with Prime Two-Day Shipping."
    },
    "2. Urgent Cancellation Request": {
        "message": "I placed order #112-9876543 ten minutes ago by mistake. Can I cancel it before it ships?",
        "context": "Order placed recently."
    },
    "3. Return / Exchange Procedure": {
        "message": "How do I return a pair of boots that don't fit? Can I drop them off at Whole Foods without a box?",
        "context": ""
    },
    "4. Explicit Legal Threat (Must Escalate)": {
        "message": "Your driver threw my package over the fence and killed my dog! I am contacting my attorney and suing Amazon tomorrow morning!",
        "context": "Customer angry, threatening lawsuit."
    },
    "5. Compromised Account Alert (Must Escalate)": {
        "message": "I received an alert that someone in Russia logged into my account and changed my password. I can't log in!",
        "context": "Suspicious login attempt."
    },
    "6. Fraudulent Card Overcharge (Must Escalate)": {
        "message": "Someone just charged $650 in gift cards to my Amazon stored card. FRAUD! Help me immediately!",
        "context": "Active unauthorized transactions."
    }
}

selected_preset_name = st.sidebar.selectbox("Choose a scenario:", list(PRESETS.keys()))
preset_data = PRESETS[selected_preset_name]

# Main inputs
col1, col2 = st.columns([2, 1])

with col1:
    customer_message = st.text_area(
        "Customer Inquiry / Tweet:",
        value=preset_data["message"],
        height=120,
        placeholder="Type customer message here..."
    )

with col2:
    conversation_context = st.text_area(
        "Optional Conversation Context:",
        value=preset_data["context"],
        height=120,
        placeholder="Preceding thread messages or background info..."
    )

process_clicked = st.button("🚀 Analyze & Generate Support Reply", type="primary", use_container_width=True)

if process_clicked and customer_message.strip():
    with st.spinner("Processing inquiry through pipeline..."):
        resp: SupportAgentResponse = pipeline.process(
            customer_message=customer_message.strip(),
            context=conversation_context.strip() if conversation_context.strip() else None
        )

    st.markdown("---")
    
    # Verdict row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Predicted Intent", value=resp.predicted_intent.replace("_", " ").title())
        st.caption(f"Intent Confidence: **{resp.intent_confidence * 100:.1f}%**")
        
    with m_col2:
        if resp.decision == "AUTO_HANDLE":
            st.success("### ✅ AUTO-HANDLE")
        else:
            st.error("### ⚠️ ESCALATE TO HUMAN")
        st.caption(f"Routing Confidence: **{resp.escalation_confidence * 100:.1f}%**")

    with m_col3:
        st.metric(label="Grounding Confidence", value=f"{resp.grounding_confidence * 100:.1f}%")
        if resp.risk_flags:
            st.warning(f"Risk Flags: {', '.join(resp.risk_flags)}")
        else:
            st.info("Risk Flags: None detected")

    # Routing Rationale
    st.markdown("#### 📋 Routing Rationale")
    st.info(f"**Reason:** {resp.escalation_reason}")
    st.caption(f"**Intent Rationale:** {resp.intent_reason}")

    # Draft Reply
    st.markdown("#### 💬 Draft Customer Support Reply")
    st.code(resp.draft_reply, language="markdown")
    st.caption(f"Character Count: {len(resp.draft_reply)} / 280 (Twitter standard)")

    # Show Evidence
    with st.expander("🔍 Show Evidence & Historical Precedents", expanded=False):
        if resp.evidence:
            st.write(f"Retrieved **{len(resp.evidence)}** historical resolutions from knowledge base:")
            for ev in resp.evidence:
                st.markdown(
                    f"**Evidence #{ev.evidence_id}** (Cosine Similarity: `{ev.similarity_score:.3f}` | Conversation: `{ev.conversation_id}`)"
                )
                st.markdown(f"- **Historical Customer:** *\"{ev.customer_message}\"*")
                st.markdown(f"- **Historical Resolution:** *\"{ev.brand_reply}\"*")
                st.markdown("---")
        else:
            st.write("No historical evidence retrieved.")

elif process_clicked:
    st.warning("Please enter a customer message first.")
