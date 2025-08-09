import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Select from 'react-select';
import { Activity, Brain, AlertCircle, CheckCircle, Loader } from 'lucide-react';

const API_BASE_URL = '/api';

function App() {
  const [symptoms, setSymptoms] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelTrained, setModelTrained] = useState(false);
  const [training, setTraining] = useState(false);

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/symptoms`);
      setSymptoms(response.data.symptoms.map(symptom => ({
        value: symptom,
        label: symptom
      })));
      setModelTrained(true);
    } catch (err) {
      setModelTrained(false);
    }
  };

  const handleTrainModel = async () => {
    setTraining(true);
    setError('');

    try {
      await axios.post(`${API_BASE_URL}/train`);
      setModelTrained(true);
      checkModelStatus();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to train model');
    } finally {
      setTraining(false);
    }
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

  const getConfidenceColor = (probability) => {
    if (probability >= 0.7) return 'bg-red-500';
    if (probability >= 0.4) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-blue-600 text-white py-6">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold text-center">Disease Prediction System</h1>
          <p className="text-center mt-2 text-blue-100">
            Select symptoms to predict possible diseases
          </p>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {!modelTrained ? (
          // Training Section
          <div className="bg-white rounded-lg shadow-md p-8 text-center">
            <Brain className="w-16 h-16 text-blue-600 mx-auto mb-4" />
            <h2 className="text-2xl font-bold mb-4">Initialize Model</h2>
            <p className="text-gray-600 mb-6">
              Train the machine learning model to start making predictions
            </p>
            
            {error && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                {error}
              </div>
            )}

            <button
              onClick={handleTrainModel}
              disabled={training}
              className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 flex items-center mx-auto"
            >
              {training ? (
                <>
                  <Loader className="w-5 h-5 animate-spin mr-2" />
                  Training...
                </>
              ) : (
                <>
                  <Brain className="w-5 h-5 mr-2" />
                  Train Model
                </>
              )}
            </button>
          </div>
        ) : (
          // Main Application
          <>
            {/* Input Section */}
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-xl font-bold mb-4">Select Your Symptoms</h2>
              
              <div className="mb-4">
                <Select
                  isMulti
                  options={symptoms}
                  value={selectedSymptoms}
                  onChange={setSelectedSymptoms}
                  placeholder="Search and select symptoms..."
                  className="mb-2"
                  isSearchable
                />
                <p className="text-sm text-gray-500">
                  Selected: {selectedSymptoms.length} symptoms
                </p>
              </div>

              {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                  <AlertCircle className="w-4 h-4 inline mr-2" />
                  {error}
                </div>
              )}

              <button
                onClick={handlePredict}
                disabled={loading || selectedSymptoms.length === 0}
                className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center"
              >
                {loading ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin mr-2" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Activity className="w-5 h-5 mr-2" />
                    Predict Diseases
                  </>
                )}
              </button>
            </div>

            {/* Results Section */}
            {predictions.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-bold mb-4">Prediction Results</h2>
                
                {/* Selected Symptoms */}
                <div className="mb-6">
                  <h3 className="font-semibold mb-2">Selected Symptoms:</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedSymptoms.map((symptom, index) => (
                      <span
                        key={index}
                        className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm"
                      >
                        {symptom.value}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Predictions */}
                <div className="space-y-4">
                  {predictions.map((prediction, index) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center">
                          <span className="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold mr-3">
                            {index + 1}
                          </span>
                          <h3 className="text-lg font-semibold">{prediction.disease}</h3>
                        </div>
                        <span className="text-xl font-bold">
                          {(prediction.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      {/* Probability Bar */}
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className={`h-3 rounded-full ${getConfidenceColor(prediction.probability)}`}
                          style={{ width: `${prediction.probability * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Disclaimer */}
                <div className="mt-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <AlertCircle className="w-5 h-5 text-yellow-600 mr-2 mt-0.5" />
                    <div>
                      <h4 className="font-semibold text-yellow-800">Medical Disclaimer</h4>
                      <p className="text-yellow-700 text-sm mt-1">
                        This is an AI prediction tool for educational purposes only. 
                        Always consult with a healthcare professional for medical advice.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;