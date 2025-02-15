import json
import pickle
import openai
import pandas as pd

with open("secrets.json", 'r') as f:
    config = json.load(f)
openai.api_key = config['openai']['api_key']

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
    'ship_type': ['Oil Service Boat'],
    'fuel_type': ['HFO'],
    'distance': [128.9],
    'fuel_consumption': [4461.45],
    'weather_conditions': ['Moderate'],
    'engine_efficiency': [92]
}
new_data_oil = {
    'ship_type': ['Oil Service Boat'],
    'fuel_type': ['Diesel'],
    'distance': [23.9],
    'weather_conditions': ['Moderate'],
    'engine_efficiency': [86.98]
}
new_data_dist = {
    'ship_type': ['Oil Service Boat'],
    'fuel_type': ['Diesel'],
    'fuel_consumption': [734.45],
    'weather_conditions': ['Calm'],
    'engine_efficiency': [3]
}

# Convert to DataFrame
new_data_co2 = pd.DataFrame(new_data)

new_data_co2[['ship_type', 'fuel_type', 'weather_conditions']] = encoder.transform(
    new_data_co2[['ship_type', 'fuel_type', 'weather_conditions']])

new_data_co2[['distance', 'fuel_consumption', 'engine_efficiency']] = scaler.transform(
    new_data_co2[['distance', 'fuel_consumption', 'engine_efficiency']].values)

new_data_fuel = pd.DataFrame(new_data_oil)
new_data_fuel[['ship_type', 'fuel_type', 'weather_conditions']] = encoder.transform(
    new_data_fuel[['ship_type', 'fuel_type', 'weather_conditions']])

new_data_fuel[['distance', 'engine_efficiency']] = scaler_fuel.transform(
    new_data_fuel[['distance', 'engine_efficiency']].values)

new_data_dist_pred = pd.DataFrame(new_data_dist)
new_data_dist_pred[['ship_type', 'fuel_type', 'weather_conditions']] = encoder.transform(
    new_data_dist_pred[['ship_type', 'fuel_type', 'weather_conditions']]
)
new_data_dist_pred[['fuel_consumption', 'engine_efficiency']] = scaler_dist.transform(
    new_data_dist_pred[['fuel_consumption', 'engine_efficiency']].values
)

# Predict using the trained model
predicted_CO2 = model.predict(new_data_co2)
predicted_fuel = model_fuel.predict(new_data_fuel)
predicted_dist = model_dist.predict(new_data_dist_pred)

"""

def recommendation_prompt(ship_data):
    prompt = (f""
            You are an assistant that recommends actions to optimize CO2 emissions Only based the provided factors.
            here is the ship data:
            -Distance Covered: {ship_data['distance']} Km
            -Fuel Consumption: {ship_data['fuel_consumption']} Liters
            -Weather Conditions {ship_data['weather_conditions']}
            -Ship Type: {ship_data['ship_type']}
            -Fuel Type: {ship_data['fuel_type']}
            -Engine Efficiency: {ship_data['engine_efficiency']}
            -Predicted CO2 Emissions {predicted_CO2[0]:.2f} Kg
            You are explaining to a non-technical user, so first tell them the provided values and the predicted CO2 emissions, and then provide one line simple recommendations
              "")

    return openai_response(prompt)


def openai_response(prompt):
    try:
        # Sending the prompt to the GPT-4 Turbo model for response
        resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Respond concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  # Limit the number of tokens
            temperature=0.3  # Controls response randomness; >0.4 for creativity, <0.4 for focus
        )

        return resp["choices"][0]["message"]["content"].strip()
    except openai.OpenAIError as e:
        return f"error: An error occurred with the AI service. Please try again later.{e}"
    except Exception as e:
        return F"An unexpected error occurred. Please try again later.{e}"


# Display the predicted results for the given input
print(recommendation_prompt(new_data))
"""
print(f"Fuel Consumption: {new_data['fuel_consumption'][0]} Liters --> Predicted CO₂ Emissions: {predicted_CO2[0]:.2f} Kg")
print(f"Distance: {new_data_oil['distance'][0]} Km --> Predicted Fuel Consumption: {predicted_fuel[0]:.2f} Liters")
print(f"Fuel Consumption: {new_data_dist['fuel_consumption'][0]} Liters --> Predicted Distance: {predicted_dist[0]:.2f} Km")

