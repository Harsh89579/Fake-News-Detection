# 📰 Fake-News-Detection

[![Machine Learning](https://img.shields.io/badge/ML-Classification-blue.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)

**Fake-News-Detection** is a machine learning project designed to identify and classify news articles as 'Real' or 'Fake' based on textual analysis. This tool addresses the growing concern of misinformation in digital media by providing an automated classification system.

## 📊 Methodology
- **Data Preprocessing**: Vectorization of text using TF-IDF and CountVectorizer.
- **Model Training**: Evaluated multiple classifiers including Logistic Regression, Passive Aggressive Classifier, and Decision Trees.
- **Evaluation**: Accuracy and Confusion Matrix analysis to ensure high precision in identifying misinformation.

## 🛠 Tech Stack
- **Languages**: Python
- **Machine Learning**: Numpy, Pandas, Scikit-learn
- **NLP**: NLTK, Regex
- **Web**: FastAPI (API), Streamlit (Frontend)

## 📂 Project Structure
```text
├── data/               # Raw and processed datasets (.csv)
├── src/                # Training scripts and core logic
├── model/              # Saved .pkl model files and weights
├── api.py              # FastAPI backend
├── app.py              # Streamlit frontend
└── requirements.txt    # Project dependencies
```

## 🚀 How to Run

### Run the API (FastAPI)
```bash
uvicorn api:app --reload
```

### Run the UI (Streamlit)
```bash
streamlit run app.py
```

---
**Author**: [Harsh Tripathi](https://github.com/Harsh89579)
