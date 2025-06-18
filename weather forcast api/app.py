from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind', 'year', 'month', 'day']

@app.route('/')
def index():
    return "🌤️ Weather Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        data = request.get_json(force=True)

        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400

        features = data['features']

        
        if len(features) != len(feature_names):
            return jsonify({'error': f'Expected {len(feature_names)} features: {feature_names}'}), 400

        
        input_array = np.array(features).reshape(1, -1)

        
        encoded_prediction = model.predict(input_array)[0]

       
        class_label = label_encoder.inverse_transform([encoded_prediction])[0]

        return jsonify({
            'predicted_class': class_label,
            'encoded_label': int(encoded_prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5050)

