from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import traceback
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
def load_models():
    try:
        nb_model = pickle.load(open('RDF_model.pkl', 'rb'))
        rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
        return nb_model, rf_model
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Load the saved scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the models
nb_model, rf_model = load_models()

# Mapping from predicted numbers to crop and fertilizer names
crop_dict = { 
    1: 'rice',
    2: 'maize',
    3: 'jute',
    4: 'cotton',
    5: 'coconut',
    6: 'papaya',
    7: 'orange',
    8: 'apple',
    9: 'muskmelon',
    10: 'watermelon',
    11: 'grapes',
    12: 'mango',
    13: 'banana',
    14: 'pomegranate',
    15: 'lentil',
    16: 'blackgram',
    17: 'mungbean',
    18: 'mothbeans',
    19: 'pigeonpeas',
    20: 'kidneybeans',
    21: 'chickpea',
    22: 'coffee'
}

fertilizer_dict = {
    0: 'Urea',
    1: 'DAP',
    2: 'Fourteen-Thirty Five-Fourteen',
    3: 'Twenty Eight-Twenty Eight',
    4: 'Seventeen-Seventeen-Seventeen',
    5: 'Twenty-Twenty',
    6: 'Ten-Twenty Six-Twenty Six'
}

@app.route('/crop_recommend', methods=['POST'])
def crop_recommend():
    print("hello i am in /crop_recommend route")
    try:
        data = request.get_json()

        if 'array' not in data:
            return jsonify({'error': "'array' key is missing from the input data"}), 400

        input_values = data['array']
        print(input_values)

        if not isinstance(input_values, list) or len(input_values) < 7:
            return jsonify({'error': "Input should be a list with at least 7 numeric elements"}), 400

        crop_features = np.array([input_values], dtype=float)
        fertilizer_features = np.array([input_values[:3]], dtype=float)

        crop_probs = nb_model.predict_proba(crop_features)[0]
        top_2_indices = np.argsort(crop_probs)[-2:][::-1]

        top_2_indices = list(dict.fromkeys(top_2_indices))

        crops = []
        preferences = ['1st preference', '2nd preference']
        for i, idx in enumerate(top_2_indices[:2]):
            crop_num = nb_model.classes_[idx]
            crop_name = crop_dict.get(crop_num, "Unknown Crop")
            crops.append({
                'name': crop_name,
                'preference': preferences[i],
                'confidence': round(crop_probs[idx] * 100, 2)
            })

        sample_data_scaled = scaler.transform(fertilizer_features)
        fertilizer_prediction = rf_model.predict(sample_data_scaled)
        predicted_fertilizer_name = fertilizer_dict.get(fertilizer_prediction[0], "Unknown Fertilizer")

        return jsonify({
            'recommended_crops': crops,
            'recommended_fertilizer': predicted_fertilizer_name
        })

    except Exception as e:
        print(f"Error occurred: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=7050)
