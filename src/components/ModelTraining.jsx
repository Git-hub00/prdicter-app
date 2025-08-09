import React, { useState } from 'react';
import axios from 'axios';
import { 
  Upload, 
  Brain, 
  Loader, 
  CheckCircle, 
  AlertCircle,
  Database,
  Zap
} from 'lucide-react';

const ModelTraining = ({ onModelTrained }) => {
  const [training, setTraining] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleTrainModel = async () => {
    setTraining(true);
    setError('');
    setSuccess('');

    try {
      const response = await axios.post('/api/train');
      setSuccess(`Model trained successfully! Accuracy: ${(response.data.accuracy * 100).toFixed(2)}%`);
      setTimeout(() => {
        onModelTrained();
      }, 2000);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to train model');
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <Brain className="w-12 h-12 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-800">Disease Prediction System</h1>
          </div>
          <p className="text-gray-600 text-lg">
            AI-powered symptom analysis for disease prediction
          </p>
        </div>

        {/* Training Card */}
        <div className="bg-white rounded-2xl card-shadow p-8">
          <div className="text-center mb-8">
            <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Database className="w-10 h-10 text-blue-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Initialize AI Model</h2>
            <p className="text-gray-600">
              Train the machine learning model with the disease-symptom dataset to start making predictions.
            </p>
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-2 gap-4 mb-8">
            <div className="flex items-center space-x-3 p-4 bg-blue-50 rounded-lg">
              <Zap className="w-6 h-6 text-blue-600" />
              <div>
                <h3 className="font-semibold text-gray-800">Fast Training</h3>
                <p className="text-sm text-gray-600">Quick model initialization</p>
              </div>
            </div>
            <div className="flex items-center space-x-3 p-4 bg-green-50 rounded-lg">
              <Brain className="w-6 h-6 text-green-600" />
              <div>
                <h3 className="font-semibold text-gray-800">Smart Predictions</h3>
                <p className="text-sm text-gray-600">AI-powered analysis</p>
              </div>
            </div>
          </div>

          {/* Messages */}
          {error && (
            <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-4 rounded-lg mb-6">
              <AlertCircle className="w-5 h-5" />
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="flex items-center space-x-2 text-green-600 bg-green-50 p-4 rounded-lg mb-6">
              <CheckCircle className="w-5 h-5" />
              <span>{success}</span>
            </div>
          )}

          {/* Train Button */}
          <button
            onClick={handleTrainModel}
            disabled={training}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-8 rounded-xl font-semibold text-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center space-x-3"
          >
            {training ? (
              <>
                <Loader className="w-6 h-6 animate-spin" />
                <span>Training Model...</span>
              </>
            ) : (
              <>
                <Upload className="w-6 h-6" />
                <span>Train AI Model</span>
              </>
            )}
          </button>

          <p className="text-center text-sm text-gray-500 mt-4">
            This process will analyze the dataset and create a machine learning model for disease prediction.
          </p>
        </div>

        {/* Info */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>The system will automatically detect and process the uploaded dataset.</p>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining;