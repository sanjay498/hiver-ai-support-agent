"""
Hiver AI Customer Support Agent — Demonstration UI.
Allows evaluators and customer support managers to:
1. Enter customer messages and conversation context.
2. View predicted intent, confidence score, and categorization reason.
3. Review AUTO-HANDLE vs ESCALATE routing decisions with safety risk flags.
4. Inspect draft customer support replies grounded in historical brand resolutions.
5. Explore the 'Show Evidence' drawer for retrieved precedent cases.
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
    layout="wide"
)


@st.cache_resource(show_spinner="Initializing Hiver AI support pipeline & FAISS index...")
def get_pipeline():
    return CustomerSupportPipeline()


pipeline = get_pipeline()
selected_brand = os.getenv("SELECTED_BRAND", "AmazonHelp")

# Header
st.title("🤖 Hiver AI Customer Support Agent")
st.caption(
    f"Autonomous customer support triage, grounded reply generation, and safe escalation engine. "
    f"Active Historical Precedent Knowledge Base: **@{selected_brand}** (Empirically selected from Kaggle Customer Support on Twitter)."
)

# Sidebar Configuration & Quick Test Presets
st.sidebar.header("⚙️ Configuration & Presets")
st.sidebar.info(
    f"**Configured Brand:** `@{selected_brand}`\n\n"
    "Discovered Taxonomy: **12 Intents**\n\n"
    "Vector Database: **FAISS IndexFlatIP (1,708 Precedents)**\n\n"
    "Evaluation Guarantee: **Strict Zero-Leakage**"
)

PRESETS = {
    "Select a pre-loaded test case...": {
        "message": "",
        "context": ""
    },
    "1. Routine Tracking Delay (Safe Auto-Handle)": {
        "message": "Where is my package? The tracking link says delivered 2 hours ago but nothing is on my porch.",
        "context": "Ordered 3 days ago with expedited shipping."
    },
    "2. Urgent Pre-Dispatch Cancellation (Safe Auto-Handle)": {
        "message": "I placed order #112-9876543 ten minutes ago by mistake. Can I cancel it before it ships?",
        "context": "Order placed recently."
    },
    "3. Return / Exchange Procedure (Safe Auto-Handle)": {
        "message": "How do I return a pair of boots that don't fit? Can I drop them off at a return location without printing a label?",
        "context": ""
    },
    "4. Explicit Legal Threat (Immediate Escalation)": {
        "message": "Your delivery driver threw my package over the fence and damaged my laptop! I am contacting my attorney and suing your company tomorrow morning!",
        "context": "Customer angry, threatening lawsuit."
    },
    "5. Compromised Account Alert (Strict Escalation)": {
        "message": "I received an alert that someone in another country logged into my account and changed my password. I am locked out!",
        "context": "Suspicious login attempt."
    },
    "6. Fraudulent Unauthorized Overcharge (Strict Escalation)": {
        "message": "Someone just charged $650 in digital gift cards to my stored card without my permission. FRAUD! Help me immediately!",
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
        placeholder="Enter incoming customer message..."
    )

with col2:
    conversation_context = st.text_area(
        "Optional Conversation Context:",
        value=preset_data["context"],
        height=120,
        placeholder="Preceding conversation thread or background context..."
    )

process_clicked = st.button("🚀 Analyze & Generate Support Resolution", type="primary", use_container_width=True)

if process_clicked and customer_message.strip():
    with st.spinner("Processing inquiry through Hiver AI pipeline..."):
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
            st.error("### ⚠️ ESCALATE TO HUMAN AGENT")
        st.caption(f"Routing Confidence: **{resp.escalation_confidence * 100:.1f}%**")

    with m_col3:
        st.metric(label="Grounding Confidence", value=f"{resp.grounding_confidence * 100:.1f}%")
        if resp.risk_flags:
            st.warning(f"Risk Flags: {', '.join(resp.risk_flags)}")
        else:
            st.info("Risk Flags: None detected (Routine)")

    # Routing Rationale
    st.markdown("#### 📋 Operational Routing Decision")
    st.info(f"**Escalation Rationale:** {resp.escalation_reason}")
    st.caption(f"**Intent Rationale:** {resp.intent_reason}")

    # Draft Reply
    st.markdown("#### 💬 Grounded Customer Support Draft")
    st.code(resp.draft_reply, language="markdown")
    st.caption(f"Character Count: {len(resp.draft_reply)} / 280 characters")

    # Show Evidence
    with st.expander("🔍 Show Evidence & Historical Precedents", expanded=False):
        if resp.evidence:
            st.write(f"Retrieved **{len(resp.evidence)}** historical resolutions from knowledge base:")
            for ev in resp.evidence:
                st.markdown(
                    f"**Evidence #{ev.evidence_id}** (Cosine Similarity: `{ev.similarity_score:.3f}` | Thread: `{ev.conversation_id}`)"
                )
                st.markdown(f"- **Historical Customer:** *\"{ev.customer_message}\"*")
                st.markdown(f"- **Historical Support Resolution:** *\"{ev.brand_reply}\"*")
                st.markdown("---")
        else:
            st.write("No historical evidence retrieved.")

elif process_clicked:
    st.warning("Please enter a customer message first.")
