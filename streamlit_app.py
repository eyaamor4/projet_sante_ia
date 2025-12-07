# streamlit_app.py
import streamlit as st
from agents.global_assistant import run_assistant
import tempfile
from PIL import Image

st.set_page_config(
    page_title="Assistant IA Santé Numérique",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Assistant Santé Numérique Agentique")

tab1, tab2 = st.tabs(["🩻 Diagnostic Image", "💬 Questions Santé Numérique"])

# ------------------------------------------------------
# 🩻 Onglet 1 : Analyse d'image avec le graph multi-agents
# ------------------------------------------------------
with tab1:
    st.header("Diagnostic par Radiographie")

    uploaded_file = st.file_uploader(
        "📤 Importer une radiographie (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image importée", use_column_width=True)

        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            temp_path = tmp.name

        if st.button("Analyser l'image 🧠"):
            st.info("⏳ Analyse en cours…")
            report = run_assistant(
                user_message="Analyse cette radiographie.",
                image_path=temp_path
            )
            st.success("✅ Analyse terminée !")
            st.markdown(report)

# ------------------------------------------------------
# 💬 Onglet 2 : Questions RAG
# ------------------------------------------------------
with tab2:
    st.header("Questions générales en santé numérique")

    user_q = st.text_input("Pose ta question (ex : Qu'est-ce que la télémédecine ?)")

    if st.button("Envoyer la question") and user_q.strip():
        st.info("⏳ Recherche d'informations…")
        answer = run_assistant(user_message=user_q)
        st.success("📄 Réponse :")
        st.markdown(answer)
