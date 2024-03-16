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
    try:
        # Convert string columns to lowercase for case-insensitive comparison
        yield_data_lower = yield_data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

        # Filter yield data based on input features
        filtered_data = yield_data_lower[
            (yield_data_lower['Area'] == area.lower()) &
            (yield_data_lower['Item'] == item.lower()) &
            (yield_data_lower['Year'] == year) &
            (yield_data_lower['average_rain_fall_mm_per_year'] == avg_rainfall) &
            (yield_data_lower['pesticides_tonnes'] == pesticides_tonnes) &
            (yield_data_lower['avg_temp'] == avg_temp)
        ]
        
        if not filtered_data.empty:
            predicted_yield = filtered_data['hg/ha_yield'].values[0]
        else:
            predicted_yield = "No data available for the specified parameters."
            
        return predicted_yield
    except Exception as e:
        print("Error occurred during yield prediction:", e)
        return "Error occurred during yield prediction. Please check the input parameters."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.form
        message = data.get('message')

        if message:
            if "hello" in message.lower() or "hi" in message.lower():
                response = "Hello! I'm here to help you with crop recommendations and yield predictions. What can I assist you with today?\n1. Recommend a crop\n2. Predict yield\nPlease enter the number corresponding to your choice."
            elif message == "1":
                response = "Sure! To recommend a crop, please provide the levels of nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, and rainfall in your area. You can enter the values like this: n=?, p=?, k=?, temperature=?, humidity=?, ph=?, rainfall=?"
            elif message == "2":
                response = "To predict yield, please specify the area of land, crop variety, cultivation year, average rainfall, pesticides usage, and average temperature. You can enter the values like this: area=?, item=?, year=?, avg_rainfall=?, pesticides_tonnes=?, avg_temp=?"
            elif all(param in message.lower() for param in ['n=', 'p=', 'k=', 'temperature=', 'humidity=', 'ph=', 'rainfall=']):
                # Extract values from the message
                values = {param.split('=')[0]: float(param.split('=')[1]) for param in message.lower().split(',')}
                recommended_crop = recommend_crop(**values)
                response = f"The recommended crop for your soil and climate conditions is {recommended_crop}. Would you like to predict the yield for this crop? If yes, please enter the required information as prompted."
            elif all(param in message.lower() for param in ['area=', 'item=', 'year=', 'avg_rainfall=', 'pesticides_tonnes=', 'avg_temp=']):
                # Extract values from the message
                values = {}
                for param in message.split(','):
                    key, value = param.split('=')
                    if key.strip() in ['area', 'item']:
                        values[key.strip()] = value.strip()
                    else:
                        values[key.strip()] = float(value.strip()) if key.strip() != 'year' else int(value.strip())
        
                predicted_yield = predict_yield(values['area'], values['item'], values['year'], values['avg_rainfall'], values['pesticides_tonnes'], values['avg_temp'])
                response = f"The predicted yield for your crop is {predicted_yield} tonnes per hectare."
            else:
                response = "I'm sorry, I didn't understand that. How can I assist you?"
        else:
            response = "I'm sorry, I didn't receive any message."

        return jsonify({'response': response})
    except Exception as e:
        print("Error occurred during request processing:", e)
        return jsonify({'response': "Error occurred during request processing. Please try again later."})

if __name__ == '__main__':
    app.run(debug=True)
