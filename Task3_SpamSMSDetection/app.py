from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    # Convert to lowercase
    lower_message = message.lower()

    # Custom spam keywords
    spam_keywords = [
        "free",
        "winner",
        "won",
        "prize",
        "cash",
        "urgent",
        "click",
        "offer",
        "limited time",
        "bank account",
        "verify",
        "recharge"
    ]

    # Rule-based spam detection
    for word in spam_keywords:
        if word in lower_message:

            result = "SPAM MESSAGE 🚨"

            return render_template(
                "index.html",
                prediction=result,
                message=message,
                confidence=99.99
            )

    # ML prediction
    transformed_message = vectorizer.transform([message])

    prediction = model.predict(transformed_message)[0]

    # Confidence score
    confidence = model.decision_function(transformed_message)[0]

    confidence_score = round(abs(confidence) * 10, 2)

    if confidence_score > 100:
        confidence_score = 99.99

    # Result
    result = "SPAM MESSAGE 🚨" if prediction == 1 else "NOT SPAM ✅"

    return render_template(
        "index.html",
        prediction=result,
        message=message,
        confidence=confidence_score
    )

if __name__ == "__main__":
    app.run(debug=True)