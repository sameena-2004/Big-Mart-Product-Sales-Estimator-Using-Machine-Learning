from flask import Flask, render_template, request
import numpy as np
import pickle
import shap

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

# SHAP explainer
explainer = shap.Explainer(model)

# ✅ FIXED LINE: convert baseline to scalar
baseline_sales = float(explainer.expected_value)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    mrp = float(request.form['mrp'])
    weight = float(request.form['weight'])
    visibility = float(request.form['visibility'])
    outlet_type = int(request.form['outlet_type'])
    location = int(request.form['location'])
    season = int(request.form['season'])
    demand = int(request.form['demand'])

    price_sensitivity = mrp / 150

    features = np.array([[ 
        mrp,
        weight,
        visibility,
        outlet_type,
        location,
        season,
        price_sensitivity,
        demand
    ]])

    # Prediction
    predicted_sales = model.predict(features)[0]

    # Stock Risk Logic
    if predicted_sales < 1800:
        risk = "HIGH RISK (Low Demand)"
    elif predicted_sales < 3000:
        risk = "MEDIUM RISK (Moderate Demand)"
    else:
        risk = "LOW RISK (High Demand)"


    # SHAP Explanation
    feature_names = [
        "Item_MRP",
        "Item_Weight",
        "Item_Visibility",
        "Outlet_Type",
        "Outlet_Location",
        "Season_Type",
        "Price_Sensitivity",
        "Demand_Trend"
    ]

    shap_values = explainer(features)
    shap_contrib = list(zip(feature_names, shap_values.values[0]))

    top_features = sorted(
        shap_contrib,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return render_template(
        "index.html",
        baseline=round(baseline_sales, 2),
        prediction=round(predicted_sales, 2),
        risk=risk,
        features=top_features
    )

if __name__ == "__main__":
    app.run(debug=True)
