import streamlit as st
import pickle
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import nltk

# ---------------- FIX NLTK FOR STREAMLIT ----------------
@st.cache_resource
def load_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

load_nltk()

from nltk.corpus import stopwords

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection (Explainable AI)",
    page_icon="📰",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

stop_words = set(stopwords.words('english'))
feature_names = vectorizer.get_feature_names_out()

# SHAP Explainer
explainer = shap.LinearExplainer(model, vectorizer.transform(["sample"]))

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- UI ----------------
st.title("📰 Fake News Detection with Explainable AI")
st.markdown("### NLP + Logistic Regression + SHAP Explainability")

input_text = st.text_area("✍️ Enter News Article:", height=250)

if st.button("🔎 Analyze Article"):

    if input_text.strip() == "":
        st.warning("Please enter text.")
        st.stop()

    cleaned = clean_text(input_text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    col1, col2 = st.columns(2)

    # ---------------- CONFIDENCE ----------------
    with col1:
        st.subheader("📊 Prediction Confidence")

        st.metric("Fake Probability", f"{probability[0]*100:.2f}%")
        st.metric("Real Probability", f"{probability[1]*100:.2f}%")

        if prediction == 1:
            st.success("✅ Final Prediction: REAL News")
        else:
            st.error("❌ Final Prediction: FAKE News")

    # ---------------- BAR GRAPH ----------------
    with col2:
        st.subheader("📈 Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], probability)
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")

        st.pyplot(fig)

    # ---------------- SHAP EXPLANATION ----------------
    st.markdown("---")
    st.subheader("🔍 SHAP Explanation (Word Contribution)")

    shap_values = explainer.shap_values(vectorized)

    shap_vals = shap_values[0]
    indices = np.argsort(np.abs(shap_vals))[::-1][:10]

    words = feature_names[indices]
    impacts = shap_vals[indices]

    explanation_df = pd.DataFrame({
        "Word": words,
        "SHAP Impact": impacts
    })

    explanation_df["Influence"] = explanation_df["SHAP Impact"].apply(
        lambda x: "REAL" if x > 0 else "FAKE"
    )

    st.dataframe(explanation_df, width='stretch')

    # SHAP BAR CHART
    fig2, ax2 = plt.subplots()

    colors = ["green" if val > 0 else "red" for val in impacts]

    ax2.barh(words, impacts, color=colors)
    ax2.set_xlabel("SHAP Impact")
    ax2.set_title("Top Word Contributions")

    st.pyplot(fig2)