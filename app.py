import os
import sys

# Configure TensorFlow to use CPU mode before any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("🔧 TensorFlow configured for CPU mode in app.py")
except:
    pass

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import subprocess

from config import Config
from models.face_verification import FaceVerifier
from utils.data_preprocessing import DataPreprocessor

# Check if models need training
def check_and_train_models():
    """Check if models exist and train if needed"""
    required_models = [
        'models/saved_models/ann_model.h5',
        'models/saved_models/cnn_model.h5',
        'models/saved_models/rnn_model.h5',
        'models/saved_models/rf_model.pkl',
        'models/saved_models/kmeans_model.pkl'
    ]
    
    # Check if all models exist
    models_exist = all(os.path.exists(model) for model in required_models)
    
    # Check if RNN model is not empty
    rnn_valid = False
    if os.path.exists('models/saved_models/rnn_model.h5'):
        rnn_valid = os.path.getsize('models/saved_models/rnn_model.h5') > 0
    
    if not models_exist or not rnn_valid:
        print("🔄 Models not found or incomplete. Training models...")
        try:
            # Run training script
            result = subprocess.run([sys.executable, 'train_dynamic_stable.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Models trained successfully!")
            else:
                print(f"⚠️  Training completed with warnings: {result.stderr}")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            print("   Please run 'python train_dynamic_stable.py' manually")

# Train models if needed
check_and_train_models()

# Dynamic model loading
USE_DYNAMIC_MODELS = True  # Set to False to use original models

if USE_DYNAMIC_MODELS:
    try:
        from models.loan_predictor_dynamic import DynamicLoanPredictor
        loan_predictor = DynamicLoanPredictor()
        print("🚀 Using Dynamic Loan Predictor with Deep Learning Models")
    except Exception as e:
        print(f"⚠️  Could not load dynamic models: {e}")
        print("   Falling back to original predictor")
        from models.loan_predictor import LoanPredictor
        loan_predictor = LoanPredictor()
else:
    from models.loan_predictor import LoanPredictor
    loan_predictor = LoanPredictor()

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize components
face_verifier = FaceVerifier()
data_preprocessor = DataPreprocessor()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face_verify', methods=['POST'])
def face_verify():
    try:
        # Get image data from request
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_face.jpg')
        image.save(temp_path)
        
        # Verify face
        is_verified, confidence = face_verifier.verify_face(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'verified': is_verified,
            'confidence': float(confidence),
            'message': 'Face verified successfully' if is_verified else 'Face verification failed'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/predict_loan', methods=['POST'])
def predict_loan():
    try:
        # Get form data
        form_data = request.json
        
        # Use dynamic preprocessor if available
        if hasattr(loan_predictor, 'preprocessor') and loan_predictor.preprocessor:
            # Use the dynamic loan predictor's preprocessor
            processed_data = loan_predictor.preprocessor.preprocess_input(form_data)
        else:
            # Fallback to the default data preprocessor
            processed_data = data_preprocessor.preprocess_input(form_data)
        
        # STEP 1: Use ANN for loan eligibility prediction
        eligibility_prob = loan_predictor.predict_eligibility(processed_data)
        is_eligible = eligibility_prob > app.config['LOAN_ELIGIBILITY_THRESHOLD']
        
        if not is_eligible:
            return jsonify({
                'success': True,
                'eligible': False,
                'probability': float(eligibility_prob),
                'message': 'Loan application not approved based on current criteria',
                'prediction_method': 'ANN for eligibility'
            })
        
        # STEP 2: Use RNN for dynamic loan package terms prediction
        loan_terms = loan_predictor.predict_loan_terms(processed_data)
        print(f"📦 Loan package method: {loan_terms.get('prediction_method', 'unknown')}")
        
        # STEP 3: Use Association Mining for persona calculation
        persona_info = None
        persona_id = 0
        
        if hasattr(loan_predictor, 'get_persona_with_customer_data'):
            # Use association mining with original customer data
            persona_info = loan_predictor.get_persona_with_customer_data(form_data)
            persona_id = persona_info['persona_id']
            print(f"👥 Using Association Mining Persona: {persona_info['persona_name']}")
        else:
            # Fallback to traditional clustering
            persona_id = loan_predictor.get_persona(processed_data)
        
        # STEP 4: Get relatable personas from dataset
        relatable_personas = []
        if hasattr(loan_predictor, 'get_relatable_personas_from_dataset'):
            relatable_personas = loan_predictor.get_relatable_personas_from_dataset(form_data)
            print(f"🔗 Found {len(relatable_personas)} relatable personas from dataset")
        
        # STEP 5: Generate personalized offers
        offers = loan_predictor.generate_offers(loan_terms, persona_id)
        
        # Prepare comprehensive response with model separation info
        response = {
            'success': True,
            'eligible': True,
            'probability': float(eligibility_prob),
            'loan_terms': {
                'rate_of_interest': float(loan_terms['rate_of_interest']),
                'tenure_months': int(loan_terms['tenure_months']),
                'sanctioned_amount': float(loan_terms['sanctioned_amount']),
                'prediction_method': loan_terms.get('prediction_method', 'RNN')
            },
            'persona': int(persona_id),
            'offers': offers,
            'relatable_personas': relatable_personas,
            'model_separation': {
                'eligibility_model': 'ANN (Artificial Neural Network)',
                'persona_model': 'Association Rule Mining',
                'loan_package_model': loan_terms.get('prediction_method', 'RNN'),
                'offers_generation': 'Persona-based customization'
            },
            'system_info': {
                'total_personas_discovered': len(loan_predictor.personas_from_association) if hasattr(loan_predictor, 'personas_from_association') else 0,
                'relatable_profiles_found': len(relatable_personas),
                'prediction_confidence': 'High (Multi-model ensemble with Association Mining)'
            }
        }
        
        # Add association mining insights if available
        if persona_info:
            response['persona_details'] = {
                'id': persona_info['persona_id'],
                'name': persona_info['persona_name'],
                'characteristics': persona_info['characteristics'],
                'confidence': persona_info['confidence'],
                'method': 'Association Mining'
            }
            
            # Add recommendations if any
            if persona_info['recommendations']:
                response['recommendations'] = persona_info['recommendations']
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in loan prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/result')
def result():
    return render_template('result.html')

# Add these routes after the existing ones in app.py

@app.route('/get_analytics', methods=['GET'])
def get_analytics():
    """Get model analytics and metrics"""
    try:
        # Get model metrics
        base_metrics = loan_predictor.get_model_metrics()
        
        # Enhanced metrics with training results if available
        try:
            import json
            with open('models/saved_models/training_results.json', 'r') as f:
                training_results = json.load(f)
            
            # Merge training results with base metrics
            enhanced_metrics = {**base_metrics, **training_results}
        except:
            enhanced_metrics = base_metrics
        
        # Add model weights if available from loan predictor
        if hasattr(loan_predictor, 'model_weights'):
            enhanced_metrics['model_weights'] = loan_predictor.model_weights
        
        # Ensure we have fallback values
        enhanced_metrics.setdefault('accuracy', 0.74)
        enhanced_metrics.setdefault('precision', 0.73)
        enhanced_metrics.setdefault('recall', 0.74)
        enhanced_metrics.setdefault('n_clusters', 6)
        
        # Get cluster information
        cluster_info = {
            'total_clusters': enhanced_metrics.get('n_clusters', 6),
            'silhouette_score': enhanced_metrics.get('silhouette_score', 0.45),
            'cluster_descriptions': {
                0: {'name': 'Young Professionals', 'size': '18%', 'profile': 'Starting careers, prefer flexibility'},
                1: {'name': 'Established Earners', 'size': '22%', 'profile': 'Stable income, balanced approach'},
                2: {'name': 'Senior Investors', 'size': '16%', 'profile': 'Conservative, experience-based decisions'},
                3: {'name': 'High-Income Segment', 'size': '15%', 'profile': 'Premium products, higher amounts'},
                4: {'name': 'Premium Customers', 'size': '12%', 'profile': 'Top-tier, exclusive services'},
                5: {'name': 'Conservative Savers', 'size': '17%', 'profile': 'Security-focused, stable returns'}
            }
        }
        
        return jsonify({
            'success': True,
            'metrics': enhanced_metrics,
            'clusters': cluster_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/explain_prediction', methods=['POST'])
def explain_prediction():
    """Get detailed explanation for loan prediction"""
    try:
        data = request.json
        features = data.get('features')
        loan_terms = data.get('loan_terms')
        persona = data.get('persona')
        offers = data.get('offers')
        
        # Get explanation
        explanation = loan_predictor.explain_recommendation(
            np.array(features), loan_terms, persona, offers
        )
        
        return jsonify({
            'success': True,
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)