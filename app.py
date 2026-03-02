import streamlit as st
import pickle
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import nltk
import os
import json

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
feature_names = vectorizer.get_feature_names_out()

# ---------------- LOAD METRICS ----------------
@st.cache_resource
def load_metrics():
    with open("metrics.json", "r") as f:
        return json.load(f)

metrics = load_metrics()

# ---------------- SHOW METRICS ----------------
st.title("📰 Fake News Detection with Explainable AI")
st.markdown("### NLP + Logistic Regression + SHAP Explainability")

st.markdown("---")
st.subheader("📊 Model Performance (Test Set)")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("F1 Score", f"{metrics['f1_score']:.4f}")
col3.metric("Precision", f"{metrics['precision']:.4f}")
col4.metric("Recall", f"{metrics['recall']:.4f}")

st.markdown("---")

# ---------------- SAFE NLTK SETUP ----------------
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

nltk.download("stopwords", download_dir=nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("punkt_tab", download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

# ---------------- SHAP EXPLAINER ----------------
background = vectorizer.transform(["sample background text"])

explainer = shap.LinearExplainer(
    model,
    background,
    feature_perturbation="interventional"
)

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------- INPUT UI ----------------
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

    # ---------------- PROBABILITY GRAPH ----------------
    with col2:
        st.subheader("📈 Probability Distribution")

        fig, ax = plt.subplots()

        if prediction == 0:  # Fake predicted
            colors = ["darkred", "lightgreen"]
        else:  # Real predicted
            colors = ["lightcoral", "darkgreen"]

        ax.bar(["Fake", "Real"], probability, color=colors)

        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")

        st.pyplot(fig)

    # ---------------- SHAP EXPLANATION ----------------
    st.markdown("---")
    st.subheader("🔍 SHAP Explanation (Top Word Contributions)")

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

    st.dataframe(explanation_df, use_container_width=True)

    fig2, ax2 = plt.subplots()
    colors = ["green" if val > 0 else "red" for val in impacts]

    ax2.barh(words, impacts, color=colors)
    ax2.set_xlabel("SHAP Impact")
    ax2.set_title("Top Word Contributions")
    st.pyplot(fig2)
