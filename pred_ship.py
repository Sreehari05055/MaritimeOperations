import pickle
import pandas as pd
import numpy

with open("model/ship_co2.pkl", "rb") as file:
    model = pickle.load(file)
with open("model/ship_fuel.pkl", "rb") as file:
    model_fuel = pickle.load(file)
with open("model/ship_dist.pkl", "rb") as file:
    model_dist = pickle.load(file)

with open("model/encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("model/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
with open("model/scaler_dist.pkl", "rb") as file:
    scaler_dist = pickle.load(file)
with open("model/scaler_fuel.pkl", "rb") as file:
    scaler_fuel = pickle.load(file)

new_data = {
    'ship_type': ['Oil Service Boat'],  # Replace with a valid ship type from your data
    'fuel_type': ['Diesel'],  # Replace with a valid fuel type from your data
    'distance': [58.9],  # Distance value for the ship
    'fuel_consumption': [734.45],  # The fuel consumption you want to test
    'weather_conditions': ['Stormy']  # Replace with a valid weather condition from your data
}
new_data_oil = {
    'ship_type': ['Oil Service Boat'],  # Replace with a valid ship type from your data
    'fuel_type': ['Diesel'],  # Replace with a valid fuel type from your data
    'distance': [116.9],  # Distance value for the ship
    'weather_conditions': ['Calm']  # Replace with a valid weather condition from your data
}
new_data_dist = {
    'ship_type': ['Oil Service Boat'],  # Replace with a valid ship type from your data
    'fuel_type': ['HFO'],  # Replace with a valid fuel type from your data
    'fuel_consumption': [734.45],
    'weather_conditions': ['Calm']  # Replace with a valid weather condition from your data
}

# Convert to DataFrame

new_data_co2 = pd.DataFrame(new_data)

new_data_co2[['ship_type', 'fuel_type', 'weather_conditions']] = encoder.transform(
    new_data_co2[['ship_type', 'fuel_type', 'weather_conditions']])

new_data_co2[['distance', 'fuel_consumption']] = scaler.transform(
    new_data_co2[['distance', 'fuel_consumption']].values)


new_data_fuel = pd.DataFrame(new_data_oil)
new_data_fuel[['ship_type', 'fuel_type', 'weather_conditions']] = encoder.transform(
    new_data_fuel[['ship_type', 'fuel_type', 'weather_conditions']])

new_data_fuel[['distance']] = scaler_fuel.transform(
    new_data_fuel[['distance']].values)

new_data_dist_pred = pd.DataFrame(new_data_dist)
new_data_dist_pred[['ship_type', 'fuel_type', 'weather_conditions']] = encoder.transform(
    new_data_dist_pred[['ship_type', 'fuel_type', 'weather_conditions']]
)
new_data_dist_pred[['fuel_consumption']] = scaler_dist.transform(
    new_data_dist_pred[['fuel_consumption']].values
)

# Predict CO₂ emissions using the trained model
predicted_CO2 = model.predict(new_data_co2)
predicted_fuel = model_fuel.predict(new_data_fuel)
predicted_dist = model_dist.predict(new_data_dist_pred)

# Predict CO₂ emissions using the trained model

# Display the predicted CO₂ emissions for the given input
print(f"Fuel Consumption: {new_data['fuel_consumption'][0]} Liters --> Predicted CO₂ Emissions: {predicted_CO2[0]:.2f} Kg")
print(f"Distance: {new_data_oil['distance'][0]} Km --> Predicted Fuel Consumption: {predicted_fuel[0]:.2f} Liters")
#print(f"Fuel Consumption: {new_data_dist['fuel_consumption'][0]} Liters --> Predicted Distance: {predicted_dist[0]:.2f} Km")
