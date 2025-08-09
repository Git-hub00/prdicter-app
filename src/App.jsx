import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Select from 'react-select';
import { 
  Activity, 
  Brain, 
  AlertCircle, 
  CheckCircle, 
  Loader, 
  TrendingUp,
  Stethoscope,
  Heart
} from 'lucide-react';
import PredictionResults from './components/PredictionResults';
import ModelTraining from './components/ModelTraining';

const API_BASE_URL = '/api';

function App() {
  const [symptoms, setSymptoms] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelTrained, setModelTrained] = useState(false);
  const [loadingSymptoms, setLoadingSymptoms] = useState(true);

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      setLoadingSymptoms(true);
      const response = await axios.get(`${API_BASE_URL}/symptoms`);
      setSymptoms(response.data.symptoms.map(symptom => ({
        value: symptom,
        label: symptom
      })));
      setModelTrained(true);
      setError('');
    } catch (err) {
      console.log('Model not trained yet');
      setModelTrained(false);
    } finally {
      setLoadingSymptoms(false);
    }
  };

  const handleModelTrained = () => {
    setModelTrained(true);
    checkModelStatus();
  };

  const handlePredict = async () => {
    if (selectedSymptoms.length === 0) {
      setError('Please select at least one symptom');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const symptomValues = selectedSymptoms.map(s => s.value);
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        symptoms: symptomValues,
        top_k: 5
      });

      setPredictions(response.data.predictions);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  const customSelectStyles = {
    control: (provided, state) => ({
      ...provided,
      minHeight: '50px',
      border: state.isFocused ? '2px solid #3b82f6' : '2px solid #e5e7eb',
      borderRadius: '12px',
      boxShadow: state.isFocused ? '0 0 0 3px rgba(59, 130, 246, 0.1)' : 'none',
      '&:hover': {
        border: '2px solid #3b82f6'
      }
    }),
    multiValue: (provided) => ({
      ...provided,
      backgroundColor: '#dbeafe',
      borderRadius: '6px'
    }),
    multiValueLabel: (provided) => ({
      ...provided,
      color: '#1e40af',
      fontWeight: '500'
    }),
    multiValueRemove: (provided) => ({
      ...provided,
      color: '#1e40af',
      '&:hover': {
        backgroundColor: '#3b82f6',
        color: 'white'
      }
    })
  };

  if (!modelTrained) {
    return <ModelTraining onModelTrained={handleModelTrained} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="gradient-bg text-white py-8">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <Stethoscope className="w-10 h-10" />
            <h1 className="text-4xl font-bold">Disease Prediction System</h1>
          </div>
          <p className="text-center text-blue-100 text-lg max-w-2xl mx-auto">
            Select your symptoms and get AI-powered disease predictions with confidence scores
          </p>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Symptom Selection Card */}
          <div className="bg-white rounded-2xl card-shadow p-8 mb-8">
            <div className="flex items-center space-x-3 mb-6">
              <Brain className="w-8 h-8 text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-800">Select Your Symptoms</h2>
            </div>

            {loadingSymptoms ? (
              <div className="flex items-center justify-center py-8">
                <Loader className="w-8 h-8 animate-spin text-blue-600" />
                <span className="ml-3 text-gray-600">Loading symptoms...</span>
              </div>
            ) : (
              <>
                <div className="mb-6">
                  <Select
                    isMulti
                    options={symptoms}
                    value={selectedSymptoms}
                    onChange={setSelectedSymptoms}
                    placeholder="Search and select symptoms..."
                    styles={customSelectStyles}
                    className="text-lg"
                    isSearchable
                    closeMenuOnSelect={false}
                  />
                  <p className="text-sm text-gray-500 mt-2">
                    Available symptoms: {symptoms.length} | Selected: {selectedSymptoms.length}
                  </p>
                </div>

                {error && (
                  <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-4 rounded-lg mb-6">
                    <AlertCircle className="w-5 h-5" />
                    <span>{error}</span>
                  </div>
                )}

                <button
                  onClick={handlePredict}
                  disabled={loading || selectedSymptoms.length === 0}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-8 rounded-xl font-semibold text-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center space-x-3"
                >
                  {loading ? (
                    <>
                      <Loader className="w-6 h-6 animate-spin" />
                      <span>Analyzing Symptoms...</span>
                    </>
                  ) : (
                    <>
                      <Activity className="w-6 h-6" />
                      <span>Predict Diseases</span>
                    </>
                  )}
                </button>
              </>
            )}
          </div>

          {/* Results */}
          {predictions.length > 0 && (
            <PredictionResults 
              predictions={predictions} 
              selectedSymptoms={selectedSymptoms.map(s => s.value)} 
            />
          )}

          {/* Info Cards */}
          <div className="grid md:grid-cols-2 gap-6 mt-8">
            <div className="bg-white rounded-xl p-6 card-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <Heart className="w-8 h-8 text-red-500" />
                <h3 className="text-xl font-semibold text-gray-800">How It Works</h3>
              </div>
              <ul className="space-y-2 text-gray-600">
                <li className="flex items-start space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <span>Select symptoms from our comprehensive database</span>
                </li>
                <li className="flex items-start space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <span>AI analyzes symptom patterns using machine learning</span>
                </li>
                <li className="flex items-start space-x-2">
                  <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <span>Get ranked disease predictions with confidence scores</span>
                </li>
              </ul>
            </div>

            <div className="bg-white rounded-xl p-6 card-shadow">
              <div className="flex items-center space-x-3 mb-4">
                <AlertCircle className="w-8 h-8 text-amber-500" />
                <h3 className="text-xl font-semibold text-gray-800">Important Notice</h3>
              </div>
              <p className="text-gray-600 leading-relaxed">
                This tool is for educational purposes only and should not replace professional medical advice. 
                Always consult with a qualified healthcare provider for proper diagnosis and treatment.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;