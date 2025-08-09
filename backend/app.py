from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re

app = Flask(__name__)
CORS(app)

class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.symptom_columns = []
        self.all_symptoms = []
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the disease dataset"""
        try:
            # Try different encodings
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='latin-1')
            
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Handle different possible column names
            symptom_col = None
            disease_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'symptom' in col_lower:
                    symptom_col = col
                elif 'disease' in col_lower or 'condition' in col_lower or 'diagnosis' in col_lower:
                    disease_col = col
            
            if symptom_col is None or disease_col is None:
                # If we can't find by name, assume first two columns
                symptom_col = df.columns[0]
                disease_col = df.columns[1]
            
            print(f"Using symptom column: {symptom_col}")
            print(f"Using disease column: {disease_col}")
            
            # Clean the data
            df = df.dropna(subset=[symptom_col, disease_col])
            
            # Extract all unique symptoms
            all_symptoms_set = set()
            
            for symptoms_str in df[symptom_col]:
                if pd.notna(symptoms_str):
                    # Split by comma and clean
                    symptoms = [s.strip().lower() for s in str(symptoms_str).split(',')]
                    symptoms = [s for s in symptoms if s and s != 'nan']
                    all_symptoms_set.update(symptoms)
            
            self.all_symptoms = sorted(list(all_symptoms_set))
            print(f"Total unique symptoms found: {len(self.all_symptoms)}")
            
            # Create one-hot encoding for symptoms
            symptom_matrix = []
            diseases = []
            
            for idx, row in df.iterrows():
                symptoms_str = str(row[symptom_col])
                disease = str(row[disease_col]).strip()
                
                if pd.notna(symptoms_str) and pd.notna(disease) and disease.lower() != 'nan':
                    # Create binary vector for symptoms
                    symptom_vector = [0] * len(self.all_symptoms)
                    patient_symptoms = [s.strip().lower() for s in symptoms_str.split(',')]
                    
                    for symptom in patient_symptoms:
                        if symptom in self.all_symptoms:
                            symptom_idx = self.all_symptoms.index(symptom)
                            symptom_vector[symptom_idx] = 1
                    
                    symptom_matrix.append(symptom_vector)
                    diseases.append(disease)
            
            # Convert to DataFrame
            X = pd.DataFrame(symptom_matrix, columns=self.all_symptoms)
            y = pd.Series(diseases)
            
            print(f"Final dataset shape: X={X.shape}, y={len(y)}")
            print(f"Number of unique diseases: {y.nunique()}")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def train_model(self, csv_path):
        """Train the disease prediction model"""
        X, y = self.load_and_preprocess_data(csv_path)
        
        # Encode disease labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save model and components
        self.save_model()
        
        return accuracy
    
    def predict_diseases(self, selected_symptoms, top_k=5):
        """Predict top K diseases based on selected symptoms"""
        if not self.model or not self.label_encoder:
            raise ValueError("Model not trained or loaded")
        
        # Create symptom vector
        symptom_vector = [0] * len(self.all_symptoms)
        
        for symptom in selected_symptoms:
            symptom_lower = symptom.lower().strip()
            if symptom_lower in self.all_symptoms:
                symptom_idx = self.all_symptoms.index(symptom_lower)
                symptom_vector[symptom_idx] = 1
        
        # Get prediction probabilities
        symptom_array = np.array(symptom_vector).reshape(1, -1)
        probabilities = self.model.predict_proba(symptom_array)[0]
        
        # Get top K predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            probability = probabilities[idx]
            predictions.append({
                'disease': disease,
                'probability': float(probability)
            })
        
        return predictions
    
    def save_model(self):
        """Save the trained model and components"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/disease_model.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.all_symptoms, 'models/symptoms.pkl')
    
    def load_model(self):
        """Load the trained model and components"""
        try:
            self.model = joblib.load('models/disease_model.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.all_symptoms = joblib.load('models/symptoms.pkl')
            return True
        except:
            return False

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
            return jsonify({'error': 'No CSV file found'}), 400
        
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