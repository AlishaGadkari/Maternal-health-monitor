from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("models/diabetes_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    tips = None
    nutrition = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        blood_pressure = float(request.form["blood_pressure"])
        glucose = float(request.form["glucose"])
        insulin = float(request.form["insulin"])
        trimester = int(request.form["trimester"])

        features = np.array([[age, bmi, blood_pressure, glucose, insulin]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "High Risk" if prediction == 1 else "Low Risk"

        # Trimester Tips
        trimester_tips = {
            1: "Take folic acid, manage nausea, avoid heavy lifting.",
            2: "Monitor fetal movement, maintain balanced diet.",
            3: "Prepare birth plan, monitor blood pressure closely."
        }
        tips = trimester_tips.get(trimester)

        # Nutrition
        if bmi < 18.5:
            nutrition = "High protein diet with dairy, nuts, whole grains."
        elif bmi < 25:
            nutrition = "Balanced diet rich in iron and calcium."
        else:
            nutrition = "Low sugar, controlled carbs, high fiber diet."

    return render_template("dashboard.html",
                           result=result,
                           probability=probability,
                           tips=tips,
                           nutrition=nutrition)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)