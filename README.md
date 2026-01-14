🛒 Big Mart Product Sales Estimator Using Machine Learning
📌 Project Overview

This project is a Machine Learning–based web application that predicts product sales at Big Mart outlets using historical data and advanced regression techniques.
The trained model is deployed using Flask, allowing users to input product and outlet details and receive real-time sales predictions.

To improve transparency and trust, the project integrates SHAP (SHapley Additive Explanations) to explain how each feature influences the prediction.

🎯 Objectives

Predict Item_Outlet_Sales accurately

Perform feature engineering and data preprocessing

Train an efficient regression model using XGBoost

Deploy the model as a Flask web application

Provide model interpretability using SHAP

🧠 Machine Learning Details
🔹 Problem Type

Supervised Learning

Regression

🔹 Algorithm Used

XGBoost Regressor (XGBRegressor)

🔹 Explainability Tool

SHAP (SHapley Additive Explanations)
Used to explain feature contributions for individual predictions.

📊 Dataset Description

The dataset contains historical sales data of Big Mart products.

Key Features Used:

Item_MRP

Item_Weight

Item_Visibility

Outlet_Type (encoded)

Outlet_Location_Type (encoded)

Season_Type (engineered)

Price_Sensitivity (engineered)

Demand_Trend (engineered)

Target Variable:

Item_Outlet_Sales

⚙️ Feature Engineering

Additional features were created to improve model performance:

Price_Sensitivity = Item_MRP / Average_Item_MRP

Season_Type = Simulated seasonal effect

Demand_Trend = Simulated demand indicator (-1, 0, 1)

📁 Project Structure
Big-Mart-Product-Sales-Estimator-Using-Machine-Learning
│
├── data/
│   └── bigmart.csv
│
├── model/
│   └── model.pkl
│
├── static/
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

SHAP

Flask

HTML / CSS

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/sameena-2004/Big-Mart-Product-Sales-Estimator-Using-Machine-Learning.git

2️⃣ Navigate to the Project Folder
cd Big-Mart-Product-Sales-Estimator-Using-Machine-Learning

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Flask Application
python app.py

5️⃣ Open in Browser
http://127.0.0.1:5000/

🔍 Model Explainability with SHAP

This project uses SHAP to explain predictions made by the XGBoost regression model.

SHAP helps:

Identify the most influential features

Understand whether a feature increases or decreases predicted sales

Improve transparency and interpretability of the model

Top contributing features are displayed along with the prediction in the web interface.

⚠️ Stock Risk Classification

Based on predicted sales:

< 1800 → HIGH RISK (Low Demand)

1800 – 3000 → MEDIUM RISK (Moderate Demand)

> 3000 → LOW RISK (High Demand)

📈 Output

Predicted product sales value

Stock risk category

Top 5 SHAP feature contributions

Baseline expected sales value

📌 Future Enhancements

Add SHAP visual plots (summary & force plots)

Improve UI/UX design

Use real seasonal and demand data

Deploy application on cloud (AWS / Heroku)

📜 License

This project is developed for academic and learning purposes.

👩‍💻 Author
Sameena Pathan
Data Science Student
GitHub: https://github.com/sameena-2004
