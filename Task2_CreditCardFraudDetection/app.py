from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    amount = float(request.form["amount"])
    city_pop = float(request.form["city_pop"])
    category = int(request.form["category"])
    gender = int(request.form["gender"])

    features = np.array([[
        0,              # merchant
        category,
        amount,
        gender,
        0,              # lat
        0,              # long
        city_pop,
        0,              # job
        0,              # unix_time
        0,              # merch_lat
        0               # merch_long
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        result = "🚨 Fraudulent Transaction"
        result_class = "fraud"
    else:
        result = "✅ Legitimate Transaction"
        result_class = "safe"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=f"{probability*100:.2f}%",
        result_class=result_class
    )

if __name__ == "__main__":
    app.run(debug=True)