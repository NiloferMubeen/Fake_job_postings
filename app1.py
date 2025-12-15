from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["job_text"]
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]

    if pred == 1:
        result = "⚠️ Fraudulent Job Posting"
    else:
        result = "✅ Real / Genuine Job Posting"

    return render_template("index1.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
