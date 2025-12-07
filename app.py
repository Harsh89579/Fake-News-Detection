from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

class NewsInput(BaseModel):
    text: str

@app.post("/predict")
def predict_news(data: NewsInput):
    transformed_text = vectorizer.transform([data.text])
    prediction = model.predict(transformed_text)[0]
    return {"prediction": "Fake News" if prediction == 0 else "Real News"}
