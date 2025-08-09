from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model import DiseasePredictor

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = DiseasePredictor()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with the dataset"""
    try:
        # Look for CSV files in the current directory
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if not csv_files:
            return jsonify({'error': 'No CSV file found. Please upload a dataset.'}), 400
        
        csv_path = csv_files[0]  # Use the first CSV file found
        print(f"Training with file: {csv_path}")
        
        accuracy = predictor.train_model(csv_path)
        
        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': accuracy,
            'total_symptoms': len(predictor.all_symptoms)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Get all available symptoms"""
    try:
        if not predictor.all_symptoms:
            # Try to load existing model
            if not predictor.load_model():
                return jsonify({'error': 'Model not trained. Please train the model first.'}), 400
        
        # Format symptoms for display (capitalize first letter)
        formatted_symptoms = [symptom.title() for symptom in predictor.all_symptoms]
        
        return jsonify({
            'symptoms': formatted_symptoms,
            'total_count': len(formatted_symptoms)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_disease():
    """Predict diseases based on selected symptoms"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        selected_symptoms = data['symptoms']
        top_k = data.get('top_k', 5)
        
        if not selected_symptoms:
            return jsonify({'error': 'Please select at least one symptom'}), 400
        
        # Load model if not already loaded
        if not predictor.model:
            if not predictor.load_model():
                return jsonify({'error': 'Model not trained. Please train the model first.'}), 400
        
        predictions = predictor.predict_diseases(selected_symptoms, top_k)
        
        return jsonify({
            'predictions': predictions,
            'selected_symptoms': selected_symptoms
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to load existing model on startup
    predictor.load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)