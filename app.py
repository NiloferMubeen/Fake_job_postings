from flask import Flask, render_template, request
import pickle
import shap
import numpy as np

app = Flask(__name__)

# Load saved model parts
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
X_train_vec = pickle.load(open("xtrain.pkl", "rb"))

# Create SHAP explainer
explainer = shap.LinearExplainer(
    model,
    X_train_vec,
    feature_names=vectorizer.get_feature_names_out()
)

def explain(text):
    vec = vectorizer.transform([text])
    shap_values = explainer.shap_values(vec)[0]

    feature_names = vectorizer.get_feature_names_out()
    idx = np.argsort(np.abs(shap_values))[::-1]

    explanation = []
    for i in idx[:10]:
        explanation.append({
            "word": feature_names[i],
            "impact": float(shap_values[i])
        })
    return explanation

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["job_text"]
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    result = "⚠️ Fraudulent" if pred == 1 else "✅ Real"

    explanation = explain(text)

    return render_template("index.html", prediction=result, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)
