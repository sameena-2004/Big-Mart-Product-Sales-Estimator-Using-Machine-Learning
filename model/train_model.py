import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../data/bigmart.csv")

data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)

le = LabelEncoder()
data['Outlet_Type'] = le.fit_transform(data['Outlet_Type'])
data['Outlet_Location_Type'] = le.fit_transform(data['Outlet_Location_Type'])

np.random.seed(42)
data['Season_Type'] = np.random.choice([0, 1, 2], size=len(data))
data['Price_Sensitivity'] = data['Item_MRP'] / data['Item_MRP'].mean()
data['Demand_Trend'] = np.random.choice([-1, 0, 1], size=len(data))

features = [
    'Item_MRP',
    'Item_Weight',
    'Item_Visibility',
    'Outlet_Type',
    'Outlet_Location_Type',
    'Season_Type',
    'Price_Sensitivity',
    'Demand_Trend'
]

X = data[features]
y = data['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")
