from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# Individual student prediction
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    attendance = float(data['attendance'])
    assignment = float(data['assignment'])
    quiz = float(data['quiz'])
    study_hours = float(data['study_hours'])

    X = np.array([[attendance, assignment, quiz, study_hours]])
    predicted_score = model.predict(X)[0]

    # Performance level
    if predicted_score >= 85:
        performance_level = "Excellent"
    elif predicted_score >= 70:
        performance_level = "Good"
    elif predicted_score >= 50:
        performance_level = "Average"
    else:
        performance_level = "Poor"

    # Weak area detection
    min_val = min(attendance, assignment, quiz)
    if min_val == attendance:
        weak_area = "Attendance"
    elif min_val == assignment:
        weak_area = "Assignment"
    else:
        weak_area = "Quiz"

    # Risk level
    if predicted_score < 50:
        risk_level = "High"
    elif predicted_score < 70:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Suggestion
    suggestion = f"Focus on {weak_area} to improve score."

    return jsonify({
        "predicted_score": round(predicted_score,2),
        "performance_level": performance_level,
        "weak_area": weak_area,
        "risk_level": risk_level,
        "suggestion": suggestion
    })


# -------------------------
# Dataset analysis
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files['file']
    df = pd.read_csv(file)

    X = df[["Attendance","Assignment","Quiz","StudyHours"]]
    df["PredictedScore"] = model.predict(X)

    # Classify performance, weak area, risk
    performance_list, weak_areas, risk_levels = [], [], []
    for idx, row in df.iterrows():
        score = row["PredictedScore"]
        if score >= 85:
            performance_list.append("Excellent")
        elif score >= 70:
            performance_list.append("Good")
        elif score >= 50:
            performance_list.append("Average")
        else:
            performance_list.append("Poor")

        # Weak area
        min_val = min(row["Attendance"], row["Assignment"], row["Quiz"])
        if min_val == row["Attendance"]:
            weak_areas.append("Attendance")
        elif min_val == row["Assignment"]:
            weak_areas.append("Assignment")
        else:
            weak_areas.append("Quiz")

        # Risk level
        if score < 50:
            risk_levels.append("High")
        elif score < 70:
            risk_levels.append("Medium")
        else:
            risk_levels.append("Low")

    df["Performance"] = performance_list
    df["WeakArea"] = weak_areas
    df["RiskLevel"] = risk_levels

    # Top 5
    top_students = df.sort_values(by="PredictedScore", ascending=False).head(5)

    # Average & Poor
    average_students = df[df["Performance"]=="Average"]
    poor_students = df[df["Performance"]=="Poor"]

    return jsonify({
        "top_students": top_students.to_dict(orient="records"),
        "average_students": average_students.to_dict(orient="records"),
        "poor_students": poor_students.to_dict(orient="records"),
        "all_students": df.to_dict(orient="records")
    })


if __name__=="__main__":
    app.run(debug=True)