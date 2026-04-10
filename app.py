from flask import Flask, render_template, request
import pickle
import requests
import os
import sqlite3

app = Flask(__name__)

# Load model
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

fake_count = 0
real_count = 0
articles_cache = []


# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news TEXT,
        result TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()

init_db()


# ---------------- REASON DETECTION ----------------
def detect_reason(text):

    text_lower = text.lower()
    reasons = []

    fake_keywords = ["shocking", "breaking", "viral", "exposed", "100%", "alert"]

    if any(word in text_lower for word in fake_keywords):
        reasons.append("Contains clickbait or sensational words")

    if text.count("!") > 3:
        reasons.append("Too many exclamation marks")

    if len(text.split()) < 25:
        reasons.append("Content is too short")

    if not any(src in text_lower for src in ["bbc", "ndtv", "reuters", "cnn"]):
        reasons.append("No trusted source mentioned")

    if text.isupper():
        reasons.append("Excessive uppercase usage")

    if not reasons:
        reasons.append("No strong fake indicators detected")

    return reasons


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html", fake=fake_count, real=real_count)


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    global fake_count, real_count

    news = request.form["news"]

    vect = vectorizer.transform([news])
    prediction = model.predict(vect)[0]

    confidence = abs(model.decision_function(vect)[0])
    confidence = round(confidence * 100, 2)

    reasons = detect_reason(news)

    if prediction == 1:
        result = "Real News"
        real_count += 1
    else:
        result = "Fake News"
        fake_count += 1

    # -------- SAVE TO DATABASE --------
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history (news, result, confidence) VALUES (?, ?, ?)",
        (news, result, confidence)
    )

    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        fake=fake_count,
        real=real_count,
        reasons=reasons
    )


# ---------------- HISTORY ----------------
@app.route("/history")
def history():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT 10")
    data = cursor.fetchall()

    conn.close()

    return render_template("history.html", data=data)


# ---------------- LATEST NEWS ----------------
@app.route("/latest")
def latest():

    global articles_cache

    API_KEY = "825fc48cd42e4d69a448134984a85346"

    url = f"https://newsapi.org/v2/everything?q=india&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    articles_cache = data["articles"]

    return render_template("latest.html", articles=articles_cache)


# ---------------- NEWS DETAIL ----------------
@app.route("/news/<int:index>")
def news_detail(index):
    article = articles_cache[index]
    return render_template("news_detail.html", article=article)


# ---------------- CHATBOT ----------------
@app.route("/chat", methods=["POST"])
def chat():

    user_query = request.form["query"].lower()

    if "fake" in user_query:
        response = "This news might be fake due to exaggerated claims or lack of trusted sources."

    elif "real" in user_query:
        response = "This news may be real if supported by verified sources."

    elif "source" in user_query:
        response = "Check the source mentioned above like BBC, NDTV, Reuters."

    elif "why" in user_query:
        response = "Fake news often uses emotional language, clickbait, or misleading headlines."

    else:
        response = "Try asking: Is this fake? Why fake? Can I trust this?"

    return response


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
