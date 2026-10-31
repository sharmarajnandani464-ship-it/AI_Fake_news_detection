from flask import Flask, render_template, request, redirect
import pickle
from datetime import datetime

app = Flask(__name__)

with open("model_bundle.pkl", "rb") as file:
    bundle = pickle.load(file)

model = bundle["model"]
vectorizer = bundle["vectorizer"]

history = []


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        news = request.form.get("news", "").strip()

        if news:
            news_vector = vectorizer.transform([news])
            result = model.predict(news_vector)[0]

            if str(result) == "0":
                prediction = "REAL NEWS"
            else:
                prediction = "FAKE NEWS"

            history.append({
                "news": news,
                "prediction": prediction,
                "time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
            })

    return render_template(
        "index.html",
        prediction=prediction,
        history=history
    )


@app.route("/history")
def show_history():
    return render_template("history.html", history=history)


@app.route("/clear_history")
def clear_history():
    history.clear()
    return redirect("/history")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run()