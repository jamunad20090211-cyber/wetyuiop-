import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("agriculture_data.csv")

# Features
X = data[['soil_ph','nitrogen','phosphorus','potassium','rainfall','temperature','humidity']]

# Targets
y_yield = data['yield']
y_fertilizer = data['fertilizer']
y_water = data['water']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
_, _, yf_train, yf_test = train_test_split(X, y_fertilizer, test_size=0.2, random_state=42)
_, _, yw_train, yw_test = train_test_split(X, y_water, test_size=0.2, random_state=42)

# Models
yield_model = RandomForestRegressor(n_estimators=100)
fertilizer_model = RandomForestRegressor(n_estimators=100)
water_model = RandomForestRegressor(n_estimators=100)

# Train models
yield_model.fit(X_train, y_train)
fertilizer_model.fit(X_train, yf_train)
water_model.fit(X_train, yw_train)

print("Models trained successfully\n")

# User input
soil_ph = float(input("Enter soil pH: "))
nitrogen = float(input("Enter nitrogen level: "))
phosphorus = float(input("Enter phosphorus level: "))
potassium = float(input("Enter potassium level: "))
rainfall = float(input("Enter rainfall (mm): "))
temperature = float(input("Enter temperature (C): "))
humidity = float(input("Enter humidity (%): "))

input_data = [[soil_ph,nitrogen,phosphorus,potassium,rainfall,temperature,humidity]]

# Predictions
predicted_yield = yield_model.predict(input_data)
predicted_fertilizer = fertilizer_model.predict(input_data)
predicted_water = water_model.predict(input_data)

print("\n----- Prediction Results -----")
print("Predicted Crop Yield:", round(predicted_yield[0],2), "tons/hectare")
print("Recommended Fertilizer:", round(predicted_fertilizer[0],2), "kg/hectare")
print("Estimated Water Requirement:", round(predicted_water[0],2), "liters/day")