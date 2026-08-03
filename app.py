import streamlit as st
import joblib
import os
from scipy.sparse import hstack, csr_matrix
import numpy as np

from rag.rag_pipeline import get_rag_response

st.set_page_config(page_title="Ticket Triage & Resolution", layout="wide")
st.title("🎫 Customer Support Ticket Triage & Resolution System") 


#loading ML model
MODEL_DIR = "models"

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

#creating 3 tabs 
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🤖 Agentic AI (RAG)", "📊 Dashboard"]) 


#Tab1-----Prediction
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

            X_input = tfidf.transform([ticket_text])

            cat_pred = cat_model.predict(X_input)[0]
            pri_pred = pri_model.predict(X_input)[0]

            cat_label = encoders["category"].inverse_transform([cat_pred])[0]
            pri_label = encoders["priority"].inverse_transform([pri_pred])[0]

            col1, col2 = st.columns(2)
            col1.metric("Predicted Category", cat_label)
            col2.metric("Predicted Priority", pri_label) 


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
            f"""<div class='tableauPlaceholder' style='width:100%; height:800px;'>
                <object class='tableauViz' style='width:100%; height:100%;'>
                    <param name='embed_code_version' value='3' />
                    <param name='site_root' value='' />
                    <param name='name' value='{TABLEAU_EMBED_URL.split("/")[-2]}/{TABLEAU_EMBED_URL.split("/")[-1]}' />
                    <param name='tabs' value='no' />
                    <param name='toolbar' value='yes' />
                </object>
            </div>""",
            height=820,
        )




 