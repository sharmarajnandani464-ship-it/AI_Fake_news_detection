from flask import Flask, render_template, request, redirect
import pickle
from datetime import datetime
app = Flask(name)
with open("fake_news_model.pkl", "rb") as file: model = pickle.load(file)
with open("vectorizer.pkl", "rb") as file: vectorizer = pickle.load(file)
history = []
@app.route("/", methods=["GET", "POST"]) def home(): prediction = None
if request.method == "POST":
    news = request.form["news"]
    news_vector = vectorizer.transform([news])
    result = model.predict(news_vector)[0]

    print("MODEL PREDICTION:", result)
    print("FAKE PROBABILITY:", model.predict_proba(news_vector)[0])

    prediction = "FAKE NEWS" if int(result) == 1 else "REAL NEWS"

    history.append({
        "news": news,
        "prediction": prediction,
        "time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
    })

return render_template("index.html", prediction=prediction, history=history)
