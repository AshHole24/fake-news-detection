from flask import Flask, render_template, request
import pickle
import requests
import os

app = Flask(__name__)

model = pickle.load(open("model/fake_news_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

fake_count = 0
real_count = 0

articles_cache = []


@app.route("/")
def home():
    return render_template("index.html", fake=fake_count, real=real_count)


@app.route("/predict", methods=["POST"])
def predict():

    global fake_count, real_count

    news = request.form["news"]

    vect = vectorizer.transform([news])

    prediction = model.predict(vect)[0]

    confidence = abs(model.decision_function(vect)[0])
    confidence = round(confidence * 100, 2)

    if prediction == 1:
        result = "Real News"
        real_count += 1
    else:
        result = "Fake News"
        fake_count += 1

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        fake=fake_count,
        real=real_count
    )


@app.route("/latest")
def latest():

    global articles_cache

    API_KEY = "825fc48cd42e4d69a448134984a85346"

    url = f"https://newsapi.org/v2/everything?q=india&language=en&sortBy=publishedAt&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    articles_cache = data["articles"]

    return render_template("latest.html", articles=articles_cache)


@app.route("/news/<int:index>")
def news_detail(index):

    article = articles_cache[index]

    return render_template("news_detail.html", article=article)


if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0", port=port)
    
@app.route("/chat", methods=["POST"])
def chat():

    user_query = request.form["query"].lower()

    # simple logic-based responses
   @app.route("/chat", methods=["POST"])
def chat():

    user_query = request.form["query"].lower()

    # more dynamic responses
    if "fake" in user_query:
        response = "This news might be fake if it contains exaggerated claims or lacks a reliable source."

    elif "real" in user_query:
        response = "This news can be considered real if it comes from trusted sources and verified facts."

    elif "source" in user_query:
        response = "You can check the source mentioned above. Trusted sources include BBC, NDTV, Reuters, etc."

    elif "why" in user_query:
        response = "Fake news usually contains emotional language, misleading headlines, or no proper source."

    elif "trust" in user_query:
        response = "To trust a news article, always cross-check with multiple reliable sources."

    else:
        response = "Try asking things like: 'Is this fake?', 'Can I trust this?', or 'What is the source?'"

    return response
