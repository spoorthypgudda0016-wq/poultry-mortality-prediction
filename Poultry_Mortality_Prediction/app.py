from flask import Flask, render_template_string, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Poultry Mortality Prediction</title>

    <style>

    body{
        font-family: Arial, sans-serif;
        background: linear-gradient(to right, #56ccf2, #2f80ed);
        margin:0;
        padding:20px;
    }

    .container{
        background:white;
        padding:30px;
        border-radius:15px;
        box-shadow:0px 0px 20px rgba(0,0,0,0.2);
        width:900px;
        margin:auto;
    }

    h2,h3{
        text-align:center;
        color:#2f80ed;
    }

    label{
        font-weight:bold;
    }

    input{
        width:100%;
        padding:10px;
        margin-top:5px;
        margin-bottom:15px;
        border-radius:8px;
        border:1px solid #ccc;
        box-sizing:border-box;
    }

    button{
        width:100%;
        padding:12px;
        background:#2f80ed;
        color:white;
        border:none;
        border-radius:8px;
        font-size:16px;
        cursor:pointer;
    }

    button:hover{
        background:#1d5fd1;
    }

    .result{
        margin-top:20px;
        text-align:center;
        font-size:20px;
        font-weight:bold;
    }

    .graph{
        text-align:center;
        margin-top:25px;
    }

    .graph img{
        width:80%;
        border-radius:10px;
        box-shadow:0px 0px 10px rgba(0,0,0,0.2);
    }

    .info{
        margin-top:15px;
        font-size:15px;
    }

    </style>

</head>

<body>

<div class="container">

    <h2>🐔 AI-Based Smart Poultry Mortality Prediction System</h2>

    <p style="text-align:center;">
        Logistic Regression + XGBoost Classifier
    </p>

    <form method="POST">

        <label>Temperature</label>
        <input type="number" step="any" name="temperature" required>

        <label>Humidity</label>
        <input type="number" step="any" name="humidity" required>

        <label>CO2</label>
        <input type="number" step="any" name="co2" required>

        <label>Flock Age</label>
        <input type="number" step="any" name="flock_age" required>

        <label>Previous Mortality Rate</label>
        <input type="number" step="any" name="prev_mortality_rate" required>

        <button type="submit">Predict</button>

    </form>

    <div class="result">
        {{ result }}
    </div>

    <hr>

    <h3>Model Accuracy Comparison</h3>

    <div class="graph">
        <img src="/static/accuracy_comparison.png">
    </div>

    <h3>Feature Importance</h3>

    <div class="graph">
        <img src="/static/feature_importance.png">
    </div>

    <h3>Confusion Matrix</h3>

    <div class="graph">
        <img src="/static/confusion_matrix.png">
    </div>

    <hr>

    <h3>About the Project</h3>

    <p class="info">
        This AI-based Smart Poultry Mortality Prediction System predicts
        next-day poultry mortality events using environmental and flock-health
        parameters such as temperature, humidity, CO2 concentration,
        flock age and previous mortality rate.
    </p>

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        data = np.array([[
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["co2"]),
            float(request.form["flock_age"]),
            float(request.form["prev_mortality_rate"])
        ]])

        data = scaler.transform(data)

        probability = model.predict_proba(data)[0][1]

        if probability < 0.3:
            result = f"🟢 LOW RISK ({probability*100:.2f}%)"

        elif probability < 0.7:
            result = f"🟡 MEDIUM RISK ({probability*100:.2f}%)"

        else:
            result = f"🔴 HIGH RISK ({probability*100:.2f}%)"

    return render_template_string(
        HTML,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)