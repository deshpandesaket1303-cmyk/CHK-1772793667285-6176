from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)
CORS(app)

# Training dataset
data = {
"Attendance":[85,60,90,75,55,88,70,92,65,50],
"Assignment":[80,50,88,70,40,82,60,90,55,40],
"Quiz":[78,45,92,65,42,80,58,95,50,35],
"StudyHours":[3,1,4,2,1,3,2,4,2,1],
"FinalScore":[82,48,91,72,45,85,65,94,60,42]
}

df = pd.DataFrame(data)

X = df[["Attendance","Assignment","Quiz","StudyHours"]]
y = df["FinalScore"]

# Train model
model = LinearRegression()
model.fit(X,y)

# Calculate RMSE (for backend evaluation)
predictions = model.predict(X)
rmse = np.sqrt(mean_squared_error(y,predictions))

print("Model RMSE:",rmse)


@app.route('/')
def home():
    return render_template("dashboard.html")


# Individual prediction
@app.route('/predict',methods=['POST'])
def predict():

    data=request.json

    attendance=float(data['attendance'])
    assignment=float(data['assignment'])
    quiz=float(data['quiz'])
    hours=float(data['hours'])

    prediction=model.predict([[attendance,assignment,quiz,hours]])[0]

    if prediction>=85:
        category="Excellent"
    elif prediction>=70:
        category="Good"
    elif prediction>=50:
        category="Average"
    else:
        category="Poor"

    recommendations=[]

    if attendance<70:
        recommendations.append("Increase lecture attendance")

    if assignment<60:
        recommendations.append("Complete assignments regularly")

    if quiz<60:
        recommendations.append("Practice more quizzes")

    if hours<2:
        recommendations.append("Increase study hours")

    if len(recommendations)==0:
        recommendations.append("Excellent performance. Keep it up!")

    return jsonify({
        "predicted_score":round(prediction,2),
        "category":category,
        "recommendations":recommendations
    })


# Dataset analysis
@app.route('/analyze_dataset',methods=['POST'])
def analyze_dataset():

    file=request.files['file']
    df=pd.read_csv(file)

    X=df[["Attendance","Assignment","Quiz","StudyHours"]]

    df["PredictedScore"]=model.predict(X)

    df["Rank"]=df["PredictedScore"].rank(ascending=False)

    def risk(score):
        if score<40:
            return "High Risk"
        elif score<60:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["Risk"]=df["PredictedScore"].apply(risk)

    top=df.sort_values(by="PredictedScore",ascending=False).head(5)
    avg=df[(df["PredictedScore"]>=50) & (df["PredictedScore"]<70)]
    poor=df[df["PredictedScore"]<50]

    return jsonify({
        "top_students":top.to_dict(orient="records"),
        "avg_students":avg.to_dict(orient="records"),
        "poor_students":poor.to_dict(orient="records"),
        "all_students":df.to_dict(orient="records")
    })


if __name__=="__main__":
    app.run(debug=True)