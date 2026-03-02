# 📰 News Article Classification: Fake vs Real

## 🚀 Live Application

If deployed, add your Streamlit link here:
https://jyofekvxbrc3xce6apvxna.streamlit.app/

---

## 📌 Project Overview

The rapid spread of misinformation through online platforms has made fake news detection a critical challenge.  
This project implements a complete NLP pipeline to automatically classify news articles and provide transparent explanations for predictions.

The system performs:

- Text preprocessing
- Feature extraction using TF-IDF
- Binary classification (Fake vs Real)
- Model evaluation
- Explainable AI using SHAP
- Web deployment with Streamlit

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Preprocessing

The text preprocessing pipeline includes:

- Converting text to lowercase
- Removing punctuation
- Removing numeric characters
- Tokenization using NLTK
- Stopword removal

This ensures cleaner and more meaningful feature representation.

---

### 2️⃣ Feature Engineering

Text is transformed into numerical features using:

**TF-IDF (Term Frequency – Inverse Document Frequency)**

TF-IDF captures:
- Word importance within a document
- Word uniqueness across the dataset

---

### 3️⃣ Model Used

**Logistic Regression**

Why Logistic Regression?

- Efficient for high-dimensional text data
- Produces probability outputs
- Interpretable coefficients
- Strong baseline for text classification

---

### 4️⃣ Model Evaluation Metrics

The model is evaluated using:

- ✅ Accuracy  
- ✅ Precision  
- ✅ Recall  
- ✅ F1 Score  

These metrics are stored in `metrics.json` and displayed dynamically in the application.

---

### 5️⃣ Explainable AI (SHAP Integration)

To improve transparency, SHAP (SHapley Additive Explanations) is used.

SHAP provides:

- Word-level contribution analysis
- Direction of influence (towards Fake or Real)
- Bar visualization of top impacting words

This ensures the system is interpretable and aligned with responsible AI practices.

---

## 🖥️ Application Interface

The Streamlit web application allows users to:

- Enter a news article
- View classification result (Fake or Real)
- See probability distribution
- Examine SHAP-based word contributions
- View model performance metrics

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Feature Extraction | TF-IDF Vectorizer |
| Explainability | SHAP |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| NLP Processing | NLTK |
| Model Serialization | Pickle |
| Metrics Storage | JSON |

---

## 📂 Project Structure

News-Article-Classification-Fake-vs-Real/
│
├── app.py # Streamlit application
├── model.pkl # Trained Logistic Regression model
├── vectorizer.pkl # TF-IDF vectorizer
├── metrics.json # Model evaluation metrics
├── requirements.txt # Project dependencies
├── output
├── .gitignore
├── News_Article_Classification_Fake_vs_Real_Report
├── README.md
├── notebook.ipynb

---
