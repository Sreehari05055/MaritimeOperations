import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv("expanded_ship_fuel_efficiency.csv")

X = pd.get_dummies(df, columns=['ship_type', 'fuel_type', 'weather_conditions'])

# Define features and target
X_dist = df[['ship_type', 'fuel_type', 'fuel_consumption', 'weather_conditions']]
y_dist = df['distance']

X_fuel = df[['ship_type', 'fuel_type', 'distance', 'weather_conditions']]
y_fuel = df['fuel_consumption']

X = df[['ship_type', 'fuel_type', 'distance', 'fuel_consumption', 'weather_conditions']]
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
X_dist[['fuel_consumption']] = scaler_dist.fit_transform(X_dist[['fuel_consumption']].values)

scaler_fuel = MinMaxScaler()
X_fuel[['distance']] = scaler_fuel.fit_transform(X_fuel[['distance']].values)

scaler = MinMaxScaler()
X[['distance', 'fuel_consumption']] = scaler.fit_transform(
    X[['distance', 'fuel_consumption']].values)

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

# Predict
dist_pred = model_dist.predict(X_dist_test)

fuel_pred = model_fuel.predict(X_fuel_test)

co2_pred = model.predict(X_test)

# Evaluate
mae_dist = mean_absolute_error(y_dist_test, dist_pred)
r2_dist = r2_score(y_dist_test, dist_pred)

mae_fuel = mean_absolute_error(y_fuel_test, fuel_pred)
r2_fuel = r2_score(y_fuel_test, fuel_pred)

mae = mean_absolute_error(y_test, co2_pred)
r2 = r2_score(y_test, co2_pred)

print(r2,r2_dist,r2_fuel)

feature_importances = pd.Series(model_fuel.feature_importances_, index=X_fuel.columns)
print(feature_importances.sort_values(ascending=False))

scores = cross_val_score(model_fuel, X, y, cv=5, scoring='r2')
print("Cross-validated R² scores:", scores)
print("Mean R²:", np.mean(scores))


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
