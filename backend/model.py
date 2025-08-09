import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
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