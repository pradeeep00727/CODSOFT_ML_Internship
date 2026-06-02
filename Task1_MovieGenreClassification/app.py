from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("genre_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    description = request.form["description"]

    vector = vectorizer.transform([description])

    prediction = model.predict(vector)[0]

    return render_template(
        "index.html",
        prediction_text=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)