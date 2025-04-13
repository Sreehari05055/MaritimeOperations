import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv("ship_fuel_efficiency.csv")

X = pd.get_dummies(df, columns=['ship_type', 'fuel_type', 'weather_conditions'])

# Define features and target
X_dist = df[['ship_type', 'fuel_type', 'fuel_consumption', 'weather_conditions', 'engine_efficiency']]
y_dist = df['distance']

X_fuel = df[['ship_type', 'fuel_type', 'distance', 'weather_conditions', 'engine_efficiency']]
y_fuel = df['fuel_consumption']

X = df[['ship_type', 'fuel_type', 'distance', 'fuel_consumption', 'weather_conditions', 'engine_efficiency']]
y = df['CO2_emissions']

# Encode categorical variables using OrdinalEncoder
encoder = OrdinalEncoder()
encoder.fit(df[['ship_type', 'fuel_type', 'weather_conditions']])

for dataset in [X_dist, X_fuel, X]:
    dataset[['ship_type', 'fuel_type', 'weather_conditions']] = encoder.transform(
        dataset[['ship_type', 'fuel_type', 'weather_conditions']]
    )
# Scale numerical features
scaler_dist = MinMaxScaler()
X_dist[['fuel_consumption', 'engine_efficiency']] = scaler_dist.fit_transform(X_dist[['fuel_consumption', 'engine_efficiency']].values)

scaler_fuel = MinMaxScaler()
X_fuel[['distance', 'engine_efficiency']] = scaler_fuel.fit_transform(X_fuel[['distance', 'engine_efficiency']].values)

scaler = MinMaxScaler()
X[['distance', 'fuel_consumption', 'engine_efficiency']] = scaler.fit_transform(
    X[['distance', 'fuel_consumption', 'engine_efficiency']].values)

X_dist_train, X_dist_test, y_dist_train, y_dist_test = train_test_split(X_dist, y_dist, test_size=0.2, random_state=42)
X_fuel_train, X_fuel_test, y_fuel_train, y_fuel_test = train_test_split(X_fuel, y_fuel, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
model_dist = RandomForestRegressor(n_estimators=100, random_state=42)
model_dist.fit(X_dist_train, y_dist_train)

model_fuel = RandomForestRegressor(n_estimators=100, random_state=42)
model_fuel.fit(X_fuel_train, y_fuel_train)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
feature_importances = pd.Series(model_fuel.feature_importances_, index=X_fuel.columns)
print(feature_importances.sort_values(ascending=False))
print("\n")
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-validated R² scores for CO2 Emissions (5 fold individual):", scores)
print("Mean R² for CO2 Emissions (5 fold mean):", np.mean(scores))
print("\n")
scores = cross_val_score(model_fuel, X, y, cv=5, scoring='r2')
print("Cross-validated R² scores for Fuel Prediction (5 fold individual):", scores)
print("Mean R² for Fuel Prediction (5 fold mean):", np.mean(scores))
print("\n")
scores = cross_val_score(model_dist, X, y, cv=5, scoring='r2')
print("Cross-validated R² scores for Distance Prediction (5 fold individual):", scores)
print("Mean R² for Distance Prediction (5 fold mean):", np.mean(scores))


with open("model/ship_dist.pkl", "wb") as file:
    pickle.dump(model_dist, file)
with open("model/ship_co2.pkl", "wb") as file:
    pickle.dump(model, file)
with open("model/ship_fuel.pkl", "wb") as file:
    pickle.dump(model_fuel, file)

with open("model/encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

with open("model/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
with open("model/scaler_dist.pkl", "wb") as file:
    pickle.dump(scaler_dist, file)
with open("model/scaler_fuel.pkl", "wb") as file:
    pickle.dump(scaler_fuel, file)

"""
X = df[['route_id', 'ship_type', 'fuel_type', 'distance', 'fuel_consumption', 'CO2_emissions', 'weather_conditions']]
y = df['engine_efficiency']

# Encode categorical variables (if not done yet)
encoder = OrdinalEncoder()
X[['route_id', 'ship_type', 'fuel_type', 'weather_conditions']] = encoder.fit_transform(
    X[['route_id', 'ship_type', 'fuel_type', 'weather_conditions']])

# Scale numerical features
scaler = MinMaxScaler()
X[['distance', 'fuel_consumption', 'CO2_emissions']] = scaler.fit_transform(
    X[['distance', 'fuel_consumption', 'CO2_emissions']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")

new_data = {
    'route_id': ['Port Harcourt-Lagos'],
    'ship_type': ['Oil Service Boat'],
    'fuel_type': ['HFO'],
    'distance': [128.9],
    'fuel_consumption': [4461.45],
    'CO2_emissions': [12779],
    'weather_conditions': ['Moderate']
}

new_data_df = pd.DataFrame(new_data)
new_data_df[['route_id', 'ship_type', 'fuel_type', 'weather_conditions']] = encoder.transform(
    new_data_df[['route_id', 'ship_type', 'fuel_type', 'weather_conditions']])
new_data_df[['distance', 'fuel_consumption', 'CO2_emissions']] = scaler.transform(
    new_data_df[['distance', 'fuel_consumption', 'CO2_emissions']])

predicted_efficiency = model.predict(new_data_df)
print(f"Predicted engine efficiency: {predicted_efficiency[0]:.2f}")
"""
