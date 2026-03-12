from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model/fake_news_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    vect = vectorizer.transform([news])
    prediction = model.predict(vect)[0]

    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)
