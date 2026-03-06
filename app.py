import streamlit as st
import pickle

model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection")
st.subheader("Enter a news headline or paragraph")

user_input = st.text_area("Type the news text here...")

if st.button("Predict"):
    if user_input:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        if prediction == 0:
            st.error("🚨 Fake News Detected!")
        else:
            st.success("✔ Real News")
    else:
        st.warning("Please enter some text before predicting.")
