import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

true_news["label"] = 1
fake_news["label"] = 0

data = pd.concat([true_news, fake_news], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

X = data["title"]  # or 'text'
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_vec))
print(f"Model accuracy: {accuracy*100:.2f}%")

pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
