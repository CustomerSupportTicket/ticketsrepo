import streamlit as st
import joblib
import os
import re
from spellchecker import SpellChecker
from scipy.sparse import hstack, csr_matrix
import numpy as np

from rag.rag_pipeline import get_rag_response

# ============================================================
# GARBAGE DETECTION SETUP
# ============================================================
# Earlier version checked whether words matched the model's own trained
# TF-IDF vocabulary — but that vocabulary is narrow (only ~5000 words
# from this specific dataset), so plenty of legitimate words ("order",
# "delayed", "delivery") were never seen during training and got
# wrongly flagged as garbage. This version instead checks against
# GENERAL English vocabulary (catches true gibberish reliably) plus a
# broad support-topic keyword list (catches real-English-but-irrelevant
# chat like "hello there" or "you dumb").
_spell = SpellChecker()

_SUPPORT_KEYWORDS = {
    "account", "login", "password", "signin", "signup", "order", "payment", "pay", "paid",
    "refund", "charge", "charged", "subscription", "subscribe", "cancel", "cancelled",
    "delivery", "deliver", "delivered", "shipping", "ship", "shipped", "bug", "crash",
    "crashed", "error", "issue", "problem", "broken", "break", "working", "slow", "lag",
    "lagging", "performance", "sync", "syncing", "data", "security", "secure", "access",
    "service", "support", "app", "application", "system", "website", "site", "feature",
    "request", "complaint", "help", "technical", "network", "connection", "connect",
    "update", "install", "download", "upload", "file", "report", "dashboard", "api",
    "server", "database", "email", "notification", "alert", "ticket", "customer",
    "product", "item", "purchase", "buy", "bought", "transaction", "invoice", "billing",
    "bill", "fail", "failed", "failure", "unable", "cannot", "cant", "missing", "lost",
    "wrong", "incorrect", "damage", "damaged", "defective", "stuck", "freeze", "frozen",
    "suspended", "suspend", "locked", "lock", "blocked", "block", "deactivated", "expired",
}


def is_garbage_ticket(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return True
    unknown = _spell.unknown(words)
    known_ratio = 1 - (len(unknown) / len(words))
    if known_ratio < 0.5:
        return True  # mostly not real English words -> gibberish
    if not any(w in _SUPPORT_KEYWORDS for w in words):
        return True  # real English, but nothing support-related -> irrelevant chat
    return False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Ticket Triage & Resolution", layout="wide")

MODEL_DIR = "models"

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
/* ---- Overall page background ---- */
.stApp {
    background-color: #F7F8FA;
}

/* ---- Centered custom title (replaces st.title) ---- */
.app-title {
    text-align: center;
    color: #1E3A5F;
    font-size: 2.4rem;
    font-weight: 800;
    padding-top: 0.5rem;
    padding-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.app-subtitle {
    text-align: center;
    color: #5C6B7A;
    font-size: 1.05rem;
    padding-bottom: 1.5rem;
}

/* ---- Center the tab list ---- */
div[role="tablist"] {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    background-color: #EAEEF2;
    border-radius: 14px;
    padding: 8px;
    margin-bottom: 1.5rem;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
}

/* ---- Bigger, styled tab buttons ---- */
div[data-testid="stTab"] {
    font-size: 1.15rem;
    font-weight: 600;
    padding: 14px 32px;
    border-radius: 10px;
    color: #445266;
    background-color: transparent;
}
div[data-testid="stTab"] p {
    font-size: 1.15rem;
    font-weight: 600;
}

/* ---- Active tab highlight ---- */
div[data-testid="stTab"][aria-selected="true"] {
    background-color: #1E3A5F;
}
div[data-testid="stTab"][aria-selected="true"] p {
    color: #FFFFFF !important;
}

/* ---- Remove default underline/selection indicator ---- */
.react-aria-SelectionIndicator {
    display: none;
}

/* ---- Accent color for buttons ---- */
.stButton>button {
    background-color: #D9822B;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.5rem;
}
.stButton>button:hover {
    background-color: #B96A1F;
    color: white;
}

/* ---- Prediction result cards ---- */
.result-row {
    display: flex;
    gap: 1.2rem;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.result-card {
    flex: 1;
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 10px rgba(30, 58, 95, 0.08);
    border-left: 6px solid #1E3A5F;
}
.result-card .result-label {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8A96A3;
    margin-bottom: 0.4rem;
}
.result-card .result-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1E3A5F;
}
.priority-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 1.3rem;
    font-weight: 800;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS FOR RESULT DISPLAY
# ============================================================
PRIORITY_COLORS = {
    "Urgent": "#D64545",   # red
    "High":   "#E08A2C",   # orange
    "Medium": "#D9B23C",   # yellow/gold
    "Low":    "#3F9142",   # green
}

def render_prediction_results(category_label, priority_label):
    priority_color = PRIORITY_COLORS.get(priority_label, "#1E3A5F")
    st.markdown(
        f"""
        <div class="result-row">
            <div class="result-card">
                <div class="result-label">Predicted Category</div>
                <div class="result-value">🏷️ {category_label}</div>
            </div>
            <div class="result-card" style="border-left-color:{priority_color};">
                <div class="result-label">Predicted Priority</div>
                <div class="priority-badge" style="background-color:{priority_color};">
                    {priority_label}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TITLE (centered, custom HTML instead of st.title)
# ============================================================
st.markdown('<div class="app-title">🎫 Customer Support Ticket Triage & Resolution System</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Predict, resolve, and track support tickets in one place</div>', unsafe_allow_html=True)

# ============================================================
# LOAD ML MODELS (cached so they load only once, not every rerun)
# ============================================================
@st.cache_resource
def load_prediction_models():
    paths = {
        "category_model": os.path.join(MODEL_DIR, "category_model.pkl"),
        "priority_model": os.path.join(MODEL_DIR, "priority_model.pkl"),
        "tfidf_vectorizer": os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
        "label_encoders": os.path.join(MODEL_DIR, "label_encoders.pkl"),
    }
    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, missing

    models = {name: joblib.load(p) for name, p in paths.items()}
    return models, []


models, missing_files = load_prediction_models()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🤖 Response", "📊 Dashboard"])

# ------------------------------------------------------------
# TAB 1 — PREDICTION
# ------------------------------------------------------------
with tab1:
    st.subheader("Predict Ticket Category & Priority")
    st.write("Enter a customer complaint below to predict its category and priority.")

    ticket_text = st.text_area(
        "Customer complaint / ticket text",
        placeholder="e.g. My payment was deducted but the order still shows as failed.",
        height=120,
        key="predict_input",
    )

    if st.button("Predict", key="predict_btn"):
        if not ticket_text.strip():
            st.warning("Please enter some ticket text first.")
        elif models is None:
            st.error(
                f"Model files not found in `{MODEL_DIR}/`. Missing: {', '.join(missing_files)}. "
                "Ask your ML teammate to export and share these .pkl files."
            )
        else:
            tfidf = models["tfidf_vectorizer"]
            cat_model = models["category_model"]
            pri_model = models["priority_model"]
            encoders = models["label_encoders"]

            if is_garbage_ticket(ticket_text):
                st.markdown(
                    """
                    <div style="background-color:#FBEAEA; border-left:6px solid #D64545;
                                border-radius:10px; padding:1.1rem 1.4rem; margin-top:1rem;">
                        <span style="font-size:1.2rem; font-weight:800; color:#B23A3A;">
                            🚫 Garbage Ticket Entered
                        </span>
                        <div style="color:#7A4444; margin-top:0.3rem; font-size:0.95rem;">
                            This doesn't look like a real customer complaint — please enter
                            an actual support ticket description.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                X_input = tfidf.transform([ticket_text])
                cat_pred = cat_model.predict(X_input)[0]
                pri_pred = pri_model.predict(X_input)[0]

                cat_label = encoders["category"].inverse_transform([cat_pred])[0]
                pri_label = encoders["priority"].inverse_transform([pri_pred])[0]

                st.success("Prediction complete ✅")

                render_prediction_results(cat_label, pri_label)

# ------------------------------------------------------------
# TAB 2 — AGENTIC AI / RAG
# ------------------------------------------------------------
with tab2:
    st.subheader("Get a Suggested Resolution (RAG)")
    st.write("Enter a customer complaint and the system will retrieve similar past "
             "resolutions and generate a suggested reply.")

    rag_input = st.text_area(
        "Customer complaint / ticket text",
        placeholder="e.g. I can't log in even though my password is correct.",
        height=120,
        key="rag_input",
    )

    if st.button("Generate Suggested Resolution", key="rag_btn"):
        if not rag_input.strip():
            st.warning("Please enter some ticket text first.")
        else:
            with st.spinner("Retrieving similar tickets and generating a response..."):
                try:
                    result = get_rag_response(rag_input)
                    st.success("Suggested Resolution")
                    st.write(result["answer"])

                    if result.get("sources"):
                        with st.expander("Retrieved reference tickets/docs"):
                            for i, src in enumerate(result["sources"], 1):
                                st.markdown(f"**{i}.** {src}")
                except NotImplementedError:
                    st.info(
                        "RAG pipeline not connected yet. Your teammate needs to "
                        "implement `get_rag_response()` in `rag/rag_pipeline.py`."
                    )
                except Exception as e:
                    st.error(f"RAG pipeline error: {e}")

# ------------------------------------------------------------
# TAB 3 — TABLEAU DASHBOARD
# ------------------------------------------------------------
with tab3:
    st.subheader("Ticket Analytics Dashboard")

    TABLEAU_EMBED_URL = "https://public.tableau.com/views/YOUR_DASHBOARD_NAME/YOUR_SHEET_NAME"

    if "YOUR_DASHBOARD_NAME" in TABLEAU_EMBED_URL:
        st.info(
            "Tableau dashboard not linked yet. Once your teammate publishes it to "
            "Tableau Public, replace `TABLEAU_EMBED_URL` in app.py with the real link."
        )
    else:
        st.components.v1.html(
            f"""
            <div class='tableauPlaceholder' style='width:100%; height:800px;'>
                <object class='tableauViz' style='width:100%; height:100%;'>
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='{TABLEAU_EMBED_URL.split("/")[-2]}/{TABLEAU_EMBED_URL.split("/")[-1]}' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='yes' />
                </object>
            </div>
            """,
            height=820,
        )