# 📰 Fake News Detection using NLP & Explainable AI  

## 🚀 Live Demo  
🔗 **Try the App Here:**  
https://jyofekvxbrc3xce6apvxna.streamlit.app/ 

---

## 📌 Project Overview  
This project is an **AI-powered Fake News Detection System** that classifies news articles as **REAL or FAKE** using **Natural Language Processing (NLP)** and **Machine Learning**.

In today’s digital world, misinformation spreads rapidly. This system helps users verify the authenticity of news content by analyzing textual patterns and linguistic features.

---

## 🎯 Key Features  
✔️ Detects Fake vs Real News  
✔️ Shows **prediction confidence (%)**  
✔️ Interactive **Streamlit Web App**  
✔️ Graphical visualization of results  
✔️ 🧠 **Explainable AI (SHAP)** – shows why the model made a decision  
✔️ Clean and user-friendly UI  

---

##  Tech Stack  

**Languages & Tools**
- Python  
- Streamlit  

**Libraries**
- Scikit-learn  
- NLTK  
- Pandas  
- NumPy  
- Matplotlib  
- SHAP  

**ML Model**
- Logistic Regression  
- TF-IDF Vectorization  

---

##  How It Works  

1️⃣ **Input**: User enters a news article  

2️⃣ **Preprocessing**:
- Lowercasing  
- Removing punctuation  
- Stopword removal  

3️⃣ **Feature Extraction**:
- Text converted into numerical vectors using TF-IDF  

4️⃣ **Prediction**:
- Logistic Regression classifies the article  

5️⃣ **Output**:
- Fake / Real prediction  
- Probability scores  
- Graph visualization  

6️⃣ **Explainability**:
- SHAP highlights top contributing words  

---

## 📊 Sample Output  

- ✅ REAL News (Confidence: 92%)  
- ❌ FAKE News (Confidence: 87%)  

With:
- Probability bar chart  
- Word-level explanation  

---

## 📂 Project Structure  

```
News-Article-Classification-Fake-vs-Real/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── Fake.csv
├── True.csv
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## 💡 Applications  

- Fake news detection platforms  
- Social media monitoring tools  
- Fact-checking systems  
- News verification apps  

## 📌 Note  
👉 Replace this with your deployed app link:  
https://your-streamlit-app-link  
