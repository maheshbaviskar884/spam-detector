from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

messages = ["Win money now", "Hello friend", "Claim free prize", "How are you"]
labels = ["spam", "ham", "spam", "ham"]

cv = CountVectorizer()
X = cv.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.form["message"]
    data = cv.transform([msg])
    result = model.predict(data)
    return render_template("index.html", result=result[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
