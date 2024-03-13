# chatbot.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load crop data
crop_data = pd.read_csv("data/crop_data.csv")

# Load yield data
yield_data = pd.read_csv("data/yield_data.csv")

# Fit Nearest Neighbors model for crop recommendation
knn = NearestNeighbors(n_neighbors=1)
knn.fit(crop_data.iloc[:, :-1])  # Exclude the last column (label)

def recommend_crop(n, p, k, temperature, humidity, ph, rainfall):
    # Predict the nearest crop based on input features
    _, idx = knn.kneighbors([[n, p, k, temperature, humidity, ph, rainfall]])
    return crop_data.iloc[idx[0]]['label'].values[0]
def predict_yield(area, item, year, avg_rainfall, pesticides_tonnes, avg_temp):
    # Filter yield data based on input features
    filtered_data = yield_data[
        (yield_data['Area'] == area) &
        (yield_data['Item'] == item) &
        (yield_data['Year'] == year) &
        (yield_data['avg_rainfall'] == avg_rainfall) &
        (yield_data['pesticides_tonnes'] == pesticides_tonnes) &
        (yield_data['avg_temp'] == avg_temp)
    ]
    if not filtered_data.empty:
        return filtered_data['Predicted_Yield'].values[0]
    else:
        return "No data available for the specified parameters."



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.form
    message = data.get('message')

    if message:
        if "recommend crop" in message.lower():
            response = "Sure! To recommend a crop, please provide the levels of nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, and rainfall in your area."
        elif "predict yield" in message.lower():
            response = "To predict yield, please specify the area of land, crop variety, cultivation year, expected yield, average rainfall, pesticides usage, and average temperature."
        elif all(param in message.lower() for param in ['n=', 'p=', 'k=', 'temperature=', 'humidity=', 'ph=', 'rainfall=']):
            # Extract values from the message
            values = {param.split('=')[0]: float(param.split('=')[1]) for param in message.lower().split(',')}
            recommended_crop = recommend_crop(**values)
            response = f"The recommended crop for your soil and climate conditions is {recommended_crop}."
        elif all(param in message.lower() for param in ['area=', 'item=', 'year=', 'hg_per_ha_yield=', 'avg_rainfall=', 'pesticides_tonnes=', 'avg_temp=']):
            # Extract values from the message
            values = {param.split('=')[0]: float(param.split('=')[1]) for param in message.lower().split(',')}
            predicted_yield = predict_yield(values['area'], values['item'], values['year'], values['hg_per_ha_yield'], values['avg_rainfall'], values['pesticides_tonnes'], values['avg_temp'])
            response = f"The predicted yield for your crop is {predicted_yield} tonnes per hectare."

        else:
            response = "I'm sorry, I didn't understand that. How can I assist you?"
    else:
        response = "I'm sorry, I didn't receive any message."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
