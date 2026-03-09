from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl","rb"))

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    attendance = float(data['attendance'])
    assignment = float(data['assignment'])
    quiz = float(data['quiz'])
    study_hours = float(data['study_hours'])

    input_data = np.array([[attendance,assignment,quiz,study_hours]])

    prediction = model.predict(input_data)

    predicted_score = round(float(prediction[0]),2)

    # Performance level
    if predicted_score >= 85:
        performance = "Excellent"
    elif predicted_score >= 70:
        performance = "Good"
    elif predicted_score >= 50:
        performance = "Average"
    else:
        performance = "Poor"

    # Weak area detection
    scores = {
        "Attendance": attendance,
        "Assignment": assignment,
        "Quiz": quiz
    }

    weak_area = min(scores, key=scores.get)

    # Risk level
    if predicted_score < 50:
        risk = "High Risk"
    elif predicted_score < 70:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    # Study suggestion
    if study_hours < 2:
        suggestion = "Increase study hours"
    elif weak_area == "Quiz":
        suggestion = "Practice more quizzes"
    elif weak_area == "Assignment":
        suggestion = "Focus on assignments"
    elif weak_area == "Attendance":
        suggestion = "Improve class attendance"
    else:
        suggestion = "Keep up the good work"

    return jsonify({
        "predicted_score": predicted_score,
        "performance_level": performance,
        "weak_area": weak_area,
        "risk_level": risk,
        "suggestion": suggestion
    })


if __name__ == "__main__":
    app.run(debug=True)