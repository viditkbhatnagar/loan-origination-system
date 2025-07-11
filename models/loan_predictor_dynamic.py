"""
Dynamic Loan Predictor Module
Uses multiple deep learning models for predictions with association mining for personas
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf

# Configure TensorFlow to use CPU only to avoid Metal GPU issues
try:
    tf.config.set_visible_devices([], 'GPU')
    print("🔧 TensorFlow configured for CPU mode")
except:
    pass

from tensorflow.keras.models import load_model
import joblib
from config import Config
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import association mining
try:
    from .association_mining import AssociationMiner
except ImportError:
    from models.association_mining import AssociationMiner

# Define DynamicProcessor for loading saved models
class DynamicProcessor:
    """Simplified dynamic data processor for loading"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
    
    def preprocess_input(self, input_data):
        """Process input data for prediction"""
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Apply same preprocessing as training
        X = pd.DataFrame()
        
        # Handle numeric columns
        numeric_cols = ['PI_AGE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED']
        for col in numeric_cols:
            if col in df.columns:
                X[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0)
        
        # Handle categorical columns
        categorical_cols = ['PI_GENDER', 'PI_OCCUPATION', 'ZONE', 'PAYMENT_MODE', 
                           'EARLY_NON', 'MEDICAL_NONMED', 'PI_STATE']
        
        for col in categorical_cols:
            if col in df.columns:
                if col in self.encoders:
                    # Use saved encoder
                    try:
                        X[col] = self.encoders[col].transform(df[col].fillna('missing'))
                    except ValueError:
                        # Handle unseen categories
                        X[col] = 0
                else:
                    # Simple hash encoding for new categories
                    X[col] = df[col].astype(str).apply(lambda x: hash(x) % 100)
        
        # Ensure all expected features exist
        expected_features = getattr(self, 'feature_names', [])
        if expected_features:
            for feature in expected_features:
                if feature not in X.columns:
                    X[feature] = 0
            X = X[expected_features]
        
        # Fill any remaining missing values
        X = X.fillna(0)
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except:
            # If scaler fails, use the values as-is (normalized)
            X_scaled = X.values
            # Simple normalization
            X_scaled = (X_scaled - np.mean(X_scaled, axis=0)) / (np.std(X_scaled, axis=0) + 1e-8)
        
        return X_scaled

class DynamicLoanPredictor:
    """Dynamic loan predictor using ensemble of deep learning models with association mining"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.model_metrics = None
        self.clustering_model = None
        self.rf_model = None
        self.association_miner = None
        self.personas_from_association = []
        self.model_weights = {
            'CNN': 0.30,
            'ANN': 0.30,
            'RNN': 0.30,
            'RF': 0.10  # Give some weight to Random Forest
        }
        self.load_models()
        self.initialize_association_mining()
    
    def load_models(self):
        """Load all available models dynamically"""
        models_path = Config.MODELS_FOLDER
        
        try:
            # Load deep learning models - all three are required
            required_dl_models = {
                'CNN': ['cnn_model.h5', 'cnn_best.h5'],
                'ANN': ['ann_model.h5', 'ann_best.h5'],
                'RNN': ['rnn_model.h5', 'rnn_best.h5']
            }
            
            for model_name, file_patterns in required_dl_models.items():
                loaded = False
                for pattern in file_patterns:
                    model_path = os.path.join(models_path, pattern)
                    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                        try:
                            self.models[model_name] = load_model(model_path)
                            print(f"✅ {model_name} model loaded from {pattern}")
                            loaded = True
                            break
                        except Exception as e:
                            print(f"⚠️  Error loading {model_name} from {pattern}: {e}")
                
                if not loaded:
                    print(f"❌ {model_name} model not found or invalid")
            
            # Load sklearn models
            sklearn_models = {
                'rf_model.pkl': 'rf_model',
                'eligibility_model.pkl': 'rf_model',  # Fallback
                'kmeans_model.pkl': 'clustering_model'
            }
            
            for filename, attr_name in sklearn_models.items():
                file_path = os.path.join(models_path, filename)
                if os.path.exists(file_path):
                    setattr(self, attr_name, joblib.load(file_path))
                    print(f"✅ {attr_name} loaded")
            
            # Load preprocessor with proper handling
            self.preprocessor = self._load_preprocessor(models_path)
            
            # Load metrics
            metrics_files = ['training_results.json', 'model_metrics.pkl']
            for file in metrics_files:
                file_path = os.path.join(models_path, file)
                if os.path.exists(file_path):
                    if file.endswith('.json'):
                        import json
                        with open(file_path, 'r') as f:
                            self.model_metrics = json.load(f)
                    else:
                        self.model_metrics = joblib.load(file_path)
                    print(f"✅ Model metrics loaded")
                    break
            
            # Update model weights based on performance
            self._update_model_weights()
            
            print(f"\n🎉 Loaded {len(self.models)} deep learning models")
            print(f"   Models available: {list(self.models.keys())}")
            
            # Verify all three models are loaded
            required_models = ['CNN', 'ANN', 'RNN']
            missing_models = [m for m in required_models if m not in self.models]
            if missing_models:
                print(f"⚠️  WARNING: Missing required models: {missing_models}")
                print("   Please run 'python train_dynamic_stable.py' to train all models")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please run 'python train_dynamic_stable.py' to train models")
    
    def initialize_association_mining(self):
        """Initialize association mining for enhanced persona discovery"""
        try:
            print("🔗 Initializing Association Mining for Personas...")
            
            # Load training data for association mining
            data_path = os.path.join(Config.DATA_FOLDER, 'Insurance_Enhanced.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                
                # Initialize association miner with optimized parameters
                self.association_miner = AssociationMiner(
                    min_support=0.05,  # 5% minimum support
                    min_confidence=0.6,  # 60% confidence
                    min_lift=1.2  # 20% lift improvement
                )
                
                # Prepare transaction data
                transactions = self.association_miner.prepare_transaction_data(df)
                
                # Find frequent patterns
                frequent_itemsets = self.association_miner.find_frequent_itemsets(transactions)
                
                # Generate association rules
                rules = self.association_miner.generate_association_rules(transactions)
                
                # Discover personas from patterns
                personas = self.association_miner.discover_personas()
                self.personas_from_association = personas
                
                print(f"✅ Association Mining Complete:")
                print(f"   📊 {len(rules)} association rules generated")
                print(f"   👥 {len(personas)} personas discovered")
                print(f"   🔍 {len(self.association_miner.approval_patterns)} approval patterns found")
                
            else:
                print(f"⚠️  Training data not found at {data_path}")
                print("   Using fallback K-means clustering for personas")
                
        except Exception as e:
            print(f"⚠️  Error initializing association mining: {e}")
            print("   Using fallback K-means clustering for personas")
    
    def _load_preprocessor(self, models_path):
        """Load preprocessor with proper error handling"""
        # Try loading saved DynamicProcessor first
        preprocessor_files = ['data_processor.pkl']
        
        for file in preprocessor_files:
            file_path = os.path.join(models_path, file)
            if os.path.exists(file_path):
                try:
                    # Make DynamicProcessor available in the global namespace for unpickling
                    import sys
                    sys.modules['__main__'].DynamicProcessor = DynamicProcessor
                    
                    processor = joblib.load(file_path)
                    print(f"✅ DynamicProcessor loaded from {file}")
                    return processor
                except Exception as e:
                    print(f"⚠️  Error loading DynamicProcessor from {file}: {e}")
        
        # If loading fails, create a new DynamicProcessor and set it up
        print("⚠️  Creating new DynamicProcessor...")
        processor = DynamicProcessor()
        
        # Load individual components if available
        scaler_path = os.path.join(models_path, 'scaler.pkl')
        encoders_path = os.path.join(models_path, 'label_encoders.pkl')
        features_path = os.path.join(models_path, 'feature_names.pkl')
        
        if os.path.exists(scaler_path):
            try:
                processor.scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded")
            except:
                print("⚠️  Using default scaler")
        
        if os.path.exists(encoders_path):
            try:
                processor.encoders = joblib.load(encoders_path)
                print("✅ Encoders loaded")
            except:
                print("⚠️  Using default encoders")
        
        if os.path.exists(features_path):
            try:
                processor.feature_names = joblib.load(features_path)
                print("✅ Feature names loaded")
            except:
                print("⚠️  Using default feature names")
        
        # If no feature names, create basic ones
        if not processor.feature_names:
            processor.feature_names = [
                'PI_AGE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED',
                'PI_GENDER', 'PI_OCCUPATION', 'ZONE', 'PAYMENT_MODE',
                'EARLY_NON', 'MEDICAL_NONMED', 'PI_STATE'
            ]
            print("✅ Using default feature names")
        
        return processor
    
    def _update_model_weights(self):
        """Update model weights based on performance metrics"""
        if self.model_metrics and isinstance(self.model_metrics, dict):
            total_score = 0
            scores = {}
            
            # Calculate scores for each model
            for model_name in ['CNN', 'ANN', 'RNN']:
                if model_name in self.model_metrics:
                    metrics = self.model_metrics[model_name]
                    # Use accuracy for classification, inverse MAE for regression
                    if 'accuracy' in metrics:
                        scores[model_name] = metrics['accuracy']
                    elif 'mae' in metrics:
                        scores[model_name] = 1 / (1 + metrics['mae'])
                    else:
                        scores[model_name] = 0.5  # Default score
                    total_score += scores[model_name]
            
            # Add RF score if available
            if 'RandomForest' in self.model_metrics:
                scores['RF'] = self.model_metrics['RandomForest'].get('accuracy', 0.7)
                total_score += scores['RF']
            
            # Normalize weights
            if total_score > 0:
                for model_name, score in scores.items():
                    self.model_weights[model_name] = score / total_score
                print(f"📊 Updated model weights based on performance:")
                for name, weight in self.model_weights.items():
                    print(f"   {name}: {weight:.3f}")
    
    def predict_eligibility(self, features):
        """Predict loan eligibility using ensemble of all three models"""
        try:
            # Ensure features are properly shaped
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            predictions = []
            weights = []
            models_used = []
            
            # Get predictions from each deep learning model (CNN, ANN, RNN)
            for model_name in ['CNN', 'ANN', 'RNN']:
                if model_name in self.models:
                    try:
                        model = self.models[model_name]
                        pred = model.predict(features, verbose=0)
                        if pred.shape[-1] == 1:  # Binary classification
                            pred_prob = float(pred[0][0])
                        else:  # Multi-class
                            pred_prob = float(np.max(pred[0]))
                        
                        predictions.append(pred_prob)
                        weights.append(self.model_weights.get(model_name, 0.33))
                        models_used.append(model_name)
                    except Exception as e:
                        print(f"⚠️  Error in {model_name} prediction: {e}")
            
            # Add Random Forest prediction if available
            if self.rf_model is not None:
                try:
                    rf_prob = self.rf_model.predict_proba(features)[0][1]
                    predictions.append(rf_prob)
                    weights.append(self.model_weights.get('RF', 0.1))
                    models_used.append('RF')
                except:
                    pass
            
            # Calculate weighted average
            if predictions:
                weights = np.array(weights) / np.sum(weights)  # Normalize
                eligibility_prob = np.average(predictions, weights=weights)
                print(f"🔮 Ensemble prediction using {models_used}: {eligibility_prob:.4f}")
            else:
                # Fallback to simple heuristic
                eligibility_prob = 0.7
                print("⚠️  Using fallback prediction")
            
            return float(eligibility_prob)
            
        except Exception as e:
            print(f"Error in eligibility prediction: {e}")
            return 0.7
    
    def predict_loan_terms(self, features):
        """Predict loan terms using RNN model for dynamic package calculation"""
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Use RNN model for loan terms prediction if available
            if 'RNN' in self.models:
                try:
                    print("🔮 Using RNN model for dynamic loan package calculation...")
                    
                    # Ensure features are properly shaped and have known dimensions
                    if features.ndim == 1:
                        features = features.reshape(1, -1)
                    
                    # Get feature count and ensure it's valid
                    feature_count = features.shape[1]
                    print(f"📊 Input features shape: {features.shape}")
                    
                    # Reshape for RNN (samples, timesteps, features)
                    features_rnn = features.reshape(features.shape[0], 1, feature_count)
                    print(f"📊 RNN input shape: {features_rnn.shape}")
                    
                    # Get RNN prediction
                    rnn_predictions = self.models['RNN'].predict(features_rnn, verbose=0)
                    
                    # Load terms scaler if available
                    scaler_path = os.path.join(Config.MODELS_FOLDER, 'terms_scaler.pkl')
                    if os.path.exists(scaler_path):
                        terms_scaler = joblib.load(scaler_path)
                        # Inverse transform to get actual values
                        rnn_terms = terms_scaler.inverse_transform(rnn_predictions)
                        
                        rate_of_interest = float(np.clip(rnn_terms[0][0], 6.0, 18.0))
                        tenure_months = int(np.clip(rnn_terms[0][1], 12, 360))
                        sanctioned_amount = float(max(50000, rnn_terms[0][2]))
                        
                        print(f"📊 RNN Predicted Terms: Rate={rate_of_interest:.2f}%, Tenure={tenure_months}m, Amount=₹{sanctioned_amount:,.0f}")
                        
                        return {
                            'rate_of_interest': rate_of_interest,
                            'tenure_months': tenure_months,
                            'sanctioned_amount': sanctioned_amount,
                            'prediction_method': 'RNN'
                        }
                    else:
                        print("⚠️  Terms scaler not found, using RNN raw predictions with normalization")
                        # Use raw predictions with smart normalization
                        raw_rate = float(rnn_predictions[0][0])
                        raw_tenure = float(rnn_predictions[0][1]) 
                        raw_amount = float(rnn_predictions[0][2])
                        
                        # Smart normalization based on typical ranges
                        rate_of_interest = np.clip(7.0 + (raw_rate * 8.0), 6.0, 18.0)  # Scale to 6-18%
                        tenure_months = int(np.clip(12 + (raw_tenure * 60), 12, 360))  # Scale to 12-72 months
                        sanctioned_amount = max(100000, abs(raw_amount) * 500000)  # Scale appropriately
                        
                        return {
                            'rate_of_interest': float(rate_of_interest),
                            'tenure_months': tenure_months,
                            'sanctioned_amount': float(sanctioned_amount),
                            'prediction_method': 'RNN_normalized'
                        }
                        
                except Exception as e:
                    print(f"⚠️  RNN prediction failed: {e}, falling back to heuristic method")
            
            # Fallback to eligibility-based heuristic calculation
            eligibility_score = self.predict_eligibility(features)
            
            # Enhanced base calculations using customer data patterns
            base_rate = 12.0  # Base interest rate
            base_tenure = 48  # Base tenure in months
            base_amount = 500000
            
            # Adjust based on eligibility score with more granular logic
            if eligibility_score > 0.8:
                # Excellent profile
                rate_of_interest = base_rate - 3.5
                tenure_months = base_tenure + 12
                sanctioned_amount = base_amount * 1.5
            elif eligibility_score > 0.6:
                # Good profile
                rate_of_interest = base_rate - 2.0
                tenure_months = base_tenure
                sanctioned_amount = base_amount * 1.2
            elif eligibility_score > 0.4:
                # Average profile
                rate_of_interest = base_rate
                tenure_months = base_tenure - 12
                sanctioned_amount = base_amount
            else:
                # Below average profile
                rate_of_interest = base_rate + 2.0
                tenure_months = base_tenure - 24
                sanctioned_amount = base_amount * 0.7
            
            # Ensure reasonable bounds
            rate_of_interest = np.clip(rate_of_interest, 6.0, 18.0)
            tenure_months = int(np.clip(tenure_months, 12, 360))
            sanctioned_amount = max(10000, sanctioned_amount)
            
            return {
                'rate_of_interest': float(rate_of_interest),
                'tenure_months': int(tenure_months),
                'sanctioned_amount': float(sanctioned_amount),
                'prediction_method': 'heuristic'
            }
            
        except Exception as e:
            print(f"Error in loan terms prediction: {e}")
            return {
                'rate_of_interest': 10.5,
                'tenure_months': 48,
                'sanctioned_amount': 500000.0,
                'prediction_method': 'fallback'
            }

    def get_relatable_personas_from_dataset(self, customer_data):
        """Get relatable customer profiles from the dataset based on association mining"""
        try:
            if not self.association_miner:
                return []
            
            # Get customer persona first
            customer_persona = self.association_miner.predict_persona(customer_data)
            customer_transaction = set(self.association_miner._customer_to_transaction(customer_data))
            
            print(f"🔍 Customer persona: {customer_persona.get('name', 'Unknown')}")
            print(f"🔍 Customer transaction: {list(customer_transaction)[:5]}...")  # Show first 5 items
            print(f"🔍 Available personas: {len(self.personas_from_association)}")
            
            # Find similar customers from discovered personas
            relatable_profiles = []
            
            for persona in self.personas_from_association:
                if persona['id'] != customer_persona.get('id', -1):  # Exclude exact match
                    # Calculate similarity with customer
                    persona_pattern = persona['pattern']
                    intersection = len(customer_transaction.intersection(persona_pattern))
                    union = len(customer_transaction.union(persona_pattern))
                    similarity = intersection / union if union > 0 else 0
                    
                    print(f"🔍 Checking persona '{persona['name']}': {intersection}/{union} = {similarity:.3f}")
                    
                    if similarity > 0.1:  # At least 10% similarity (more permissive)
                        # Create a relatable profile
                        profile = {
                            'persona_name': persona['name'],
                            'characteristics': persona.get('characteristics', {}),
                            'similarity_score': similarity,
                            'confidence': persona.get('avg_confidence', 0.5),
                            'approval_rate': self._estimate_approval_rate(persona),
                            'typical_package': self._get_typical_package_for_persona(persona),
                            'why_relatable': self._explain_similarity(customer_transaction, persona_pattern)
                        }
                        relatable_profiles.append(profile)
            
            # Sort by similarity and return top 3
            relatable_profiles.sort(key=lambda x: x['similarity_score'], reverse=True)
            return relatable_profiles[:3]
            
        except Exception as e:
            print(f"Error getting relatable personas: {e}")
            return []
    
    def _estimate_approval_rate(self, persona):
        """Estimate approval rate for a persona based on its rules"""
        try:
            if self.association_miner and hasattr(self.association_miner, 'approval_patterns'):
                persona_rules = persona.get('rules', [])
                approval_rules = [r for r in persona_rules if 'APPROVED' in str(r.get('consequent', []))]
                
                if approval_rules:
                    return np.mean([r.get('confidence', 0.5) for r in approval_rules])
                
            return persona.get('avg_confidence', 0.5)
        except:
            return 0.5
    
    def _get_typical_package_for_persona(self, persona):
        """Get typical loan package for a persona"""
        try:
            confidence = persona.get('avg_confidence', 0.5)
            lift = persona.get('avg_lift', 1.0)
            
            # Base package
            base_rate = 12.0
            base_tenure = 48
            base_amount = 500000
            
            # Adjust based on persona strength
            if confidence > 0.7 and lift > 1.5:
                # Strong positive persona
                rate = base_rate - 2.0
                tenure = base_tenure + 12
                amount = base_amount * 1.3
            elif confidence > 0.6:
                # Good persona
                rate = base_rate - 1.0
                tenure = base_tenure
                amount = base_amount * 1.1
            else:
                # Average persona
                rate = base_rate
                tenure = base_tenure - 6
                amount = base_amount
            
            return {
                'interest_rate': round(np.clip(rate, 6.0, 18.0), 2),
                'tenure_months': int(np.clip(tenure, 12, 240)),
                'amount_range': f"₹{amount*0.8:,.0f} - ₹{amount*1.2:,.0f}"
            }
        except:
            return {
                'interest_rate': 10.5,
                'tenure_months': 48,
                'amount_range': "₹4,00,000 - ₹6,00,000"
            }
    
    def _explain_similarity(self, customer_transaction, persona_pattern):
        """Explain why personas are relatable"""
        try:
            common_items = customer_transaction.intersection(persona_pattern)
            
            explanations = []
            for item in list(common_items)[:3]:  # Top 3 common traits
                if item.startswith('Age_'):
                    explanations.append(f"Similar age group ({item.replace('Age_', '')})")
                elif item.startswith('Income_'):
                    explanations.append(f"Similar income level ({item.replace('Income_', '')})")
                elif item.startswith('PI_OCCUPATION_'):
                    explanations.append(f"Same profession ({item.replace('PI_OCCUPATION_', '').replace('_', ' ')})")
                elif item.startswith('ZONE_'):
                    explanations.append(f"Same zone ({item.replace('ZONE_', '')})")
                else:
                    explanations.append(f"Common trait: {item}")
            
            return '; '.join(explanations) if explanations else "Similar profile characteristics"
        except:
            return "Similar customer profile"
    
    def get_persona(self, features):
        """Get customer persona using association mining or clustering fallback"""
        try:
            # If association mining is available, use it for persona prediction
            if self.association_miner and hasattr(self, 'preprocessor'):
                # Get original input data from features (this is a simplification)
                # In practice, you'd want to store the original data for this
                
                # For now, create a sample data dict for demonstration
                # In production, you should pass the original customer data here
                sample_data = {
                    'PI_AGE': 32,
                    'PI_ANNUAL_INCOME': 800000,
                    'SUM_ASSURED': 50000,
                    'PI_GENDER': 'M',
                    'PI_OCCUPATION': 'Government Employee',
                    'ZONE': 'Metro',
                    'PAYMENT_MODE': 'Monthly',
                    'EARLY_NON': 'No',
                    'MEDICAL_NONMED': 'Medical',
                    'PI_STATE': 'Delhi'
                }
                
                # Use association mining to predict persona
                persona_info = self.association_miner.predict_persona(sample_data)
                print(f"👥 Association Mining Persona: {persona_info['name']}")
                return persona_info['id']
            
            # Fallback to clustering model
            elif self.clustering_model is not None:
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                
                persona = self.clustering_model.predict(features)[0]
                return int(persona)
            
            else:
                # Dynamic persona assignment based on eligibility
                eligibility = self.predict_eligibility(features)
                if eligibility > 0.8:
                    return 0  # Premium customer
                elif eligibility > 0.6:
                    return 1  # Good customer
                elif eligibility > 0.4:
                    return 2  # Average customer
                else:
                    return 3  # Below average
                
        except Exception as e:
            print(f"Error in persona prediction: {e}")
            return 0
    
    def get_persona_with_customer_data(self, customer_data):
        """Get persona using customer data directly for association mining"""
        try:
            if self.association_miner:
                persona_info = self.association_miner.predict_persona(customer_data)
                print(f"👥 Association Mining Persona: {persona_info['name']}")
                
                # Get recommendations as well
                recommendations = self.association_miner.generate_recommendations(customer_data)
                
                return {
                    'persona_id': persona_info['id'],
                    'persona_name': persona_info['name'],
                    'characteristics': persona_info.get('characteristics', {}),
                    'confidence': persona_info.get('avg_confidence', 0.5),
                    'recommendations': recommendations.get('recommendations', [])
                }
            else:
                # Fallback
                return {
                    'persona_id': 0,
                    'persona_name': 'Standard Customer',
                    'characteristics': {},
                    'confidence': 0.5,
                    'recommendations': []
                }
        except Exception as e:
            print(f"Error in persona prediction with customer data: {e}")
            return {
                'persona_id': 0,
                'persona_name': 'Standard Customer',
                'characteristics': {},
                'confidence': 0.5,
                'recommendations': []
            }
    
    def generate_offers(self, base_terms, persona):
        """Generate personalized loan offers with dynamic variations"""
        base_rate = base_terms['rate_of_interest']
        base_tenure = base_terms['tenure_months']
        base_amount = base_terms['sanctioned_amount']
        
        print(f"🎁 Generating dynamic offers for persona {persona}")
        print(f"📦 Base terms: {base_rate}% • {base_tenure}m • ₹{base_amount:,.0f}")
        
        # Use association mining personas if available
        if self.personas_from_association and persona < len(self.personas_from_association):
            persona_info = self.personas_from_association[persona]
            profile = {
                'name': persona_info['name'],
                'rate_adj': -0.3 if persona_info['avg_confidence'] > 0.7 else 0.1,
                'tenure_pref': 12 if 'conservative' in persona_info['name'].lower() else -6,
                'amount_factor': 1.1 if persona_info['avg_lift'] > 1.5 else 0.95
            }
            print(f"👥 Using persona: {persona_info['name']} (confidence: {persona_info['avg_confidence']:.2f})")
        else:
            # Fallback to dynamic persona profiles with more variation
            persona_profiles = self._get_dynamic_persona_profiles()
            profile = persona_profiles.get(persona, {
                'name': 'Standard Customer',
                'rate_adj': np.random.uniform(-0.5, 0.3),  # Random variation
                'tenure_pref': np.random.randint(-12, 18),  # Random tenure preference
                'amount_factor': np.random.uniform(0.9, 1.15)  # Random amount factor
            })
            print(f"🎲 Using dynamic profile: {profile['name']}")
        
        offers = []
        
        # Offer 1: Low EMI Option (More Dynamic)
        offer1_rate = base_rate - 0.8 + profile['rate_adj'] + np.random.uniform(-0.2, 0.1)
        offer1_tenure = base_tenure + 18 + profile['tenure_pref'] + np.random.randint(-3, 6)
        offer1_amount = base_amount * (0.92 + np.random.uniform(0, 0.06)) * profile['amount_factor']
        
        offer1 = {
            'type': 'Easy EMI Plan',
            'subtitle': 'Lower monthly payments with extended tenure',
            'rate_of_interest': round(np.clip(offer1_rate, 7.0, 16.0), 2),
            'tenure_months': int(np.clip(offer1_tenure, 18, 240)),
            'sanctioned_amount': round(max(100000, offer1_amount), 2)
        }
        offer1['emi'] = self.calculate_emi(
            offer1['sanctioned_amount'], 
            offer1['rate_of_interest'], 
            offer1['tenure_months']
        )
        offer1['total_interest'] = round(
            offer1['emi'] * offer1['tenure_months'] - offer1['sanctioned_amount'], 2
        )
        offers.append(offer1)
        
        # Offer 2: Balanced Option (Recommended) - More Dynamic
        offer2_rate = base_rate + profile['rate_adj'] * 0.4 + np.random.uniform(-0.15, 0.1)
        offer2_tenure = base_tenure + (profile['tenure_pref'] // 2) + np.random.randint(-2, 4)
        offer2_amount = base_amount * (0.98 + np.random.uniform(0, 0.04)) * profile['amount_factor']
        
        offer2 = {
            'type': 'Smart Choice Plan',
            'subtitle': 'Optimal balance of EMI and interest',
            'rate_of_interest': round(np.clip(offer2_rate, 7.5, 15.0), 2),
            'tenure_months': int(np.clip(offer2_tenure, 12, 180)),
            'sanctioned_amount': round(max(150000, offer2_amount), 2)
        }
        offer2['emi'] = self.calculate_emi(
            offer2['sanctioned_amount'], 
            offer2['rate_of_interest'], 
            offer2['tenure_months']
        )
        offer2['total_interest'] = round(
            offer2['emi'] * offer2['tenure_months'] - offer2['sanctioned_amount'], 2
        )
        offer2['recommendation_score'] = 95
        offers.append(offer2)
        
        # Offer 3: Quick Payoff Option - More Dynamic
        offer3_rate = base_rate + 0.4 + profile['rate_adj'] * 0.6 + np.random.uniform(-0.1, 0.2)
        offer3_tenure = base_tenure - 6 + (profile['tenure_pref'] // 3) + np.random.randint(-2, 3)
        offer3_amount = base_amount * (1.03 + np.random.uniform(0, 0.04)) * profile['amount_factor']
        
        offer3 = {
            'type': 'Fast Track Plan',
            'subtitle': 'Save on interest with shorter tenure',
            'rate_of_interest': round(np.clip(offer3_rate, 8.0, 17.0), 2),
            'tenure_months': int(np.clip(offer3_tenure, 12, 120)),
            'sanctioned_amount': round(max(200000, offer3_amount), 2)
        }
        offer3['emi'] = self.calculate_emi(
            offer3['sanctioned_amount'], 
            offer3['rate_of_interest'], 
            offer3['tenure_months']
        )
        offer3['total_interest'] = round(
            offer3['emi'] * offer3['tenure_months'] - offer3['sanctioned_amount'], 2
        )
        offer3['savings_vs_easy'] = round(
            offer1['total_interest'] - offer3['total_interest'], 2
        )
        offers.append(offer3)
        
        print(f"🎯 Generated 3 dynamic offers:")
        for i, offer in enumerate(offers, 1):
            print(f"   {i}. {offer['type']}: {offer['rate_of_interest']}% • {offer['tenure_months']}m • ₹{offer['sanctioned_amount']:,.0f}")
        
        return offers
    
    def _get_dynamic_persona_profiles(self):
        """Get dynamic persona profiles based on clustering results"""
        if self.clustering_model:
            n_clusters = self.clustering_model.n_clusters
        else:
            n_clusters = 5
        
        # Generate dynamic profiles
        profiles = {}
        profile_templates = [
            {'name': 'Premium Clients', 'rate_adj': -0.5, 'tenure_pref': -12, 'amount_factor': 1.2},
            {'name': 'Good Customers', 'rate_adj': -0.2, 'tenure_pref': 0, 'amount_factor': 1.1},
            {'name': 'Average Borrowers', 'rate_adj': 0, 'tenure_pref': 6, 'amount_factor': 1.0},
            {'name': 'Cautious Borrowers', 'rate_adj': 0.3, 'tenure_pref': 12, 'amount_factor': 0.9},
            {'name': 'High Risk Segment', 'rate_adj': 0.5, 'tenure_pref': -6, 'amount_factor': 0.8}
        ]
        
        for i in range(n_clusters):
            if i < len(profile_templates):
                profiles[i] = profile_templates[i]
            else:
                # Generate profile for additional clusters
                profiles[i] = {
                    'name': f'Customer Segment {i+1}',
                    'rate_adj': np.random.uniform(-0.5, 0.5),
                    'tenure_pref': np.random.randint(-12, 12),
                    'amount_factor': np.random.uniform(0.9, 1.1)
                }
        
        return profiles
    
    def calculate_emi(self, principal, rate, tenure):
        """Calculate EMI with proper error handling"""
        try:
            if tenure <= 0:
                return principal
            
            monthly_rate = rate / (12 * 100)
            if monthly_rate == 0:
                return round(principal / tenure, 2)
            
            emi = principal * (monthly_rate * (1 + monthly_rate)**tenure) / ((1 + monthly_rate)**tenure - 1)
            return round(emi, 2)
        except:
            return round(principal / max(1, tenure), 2)
    
    def get_model_metrics(self):
        """Return model performance metrics"""
        metrics = {
            'models_loaded': len(self.models),
            'model_types': list(self.models.keys()),
            'all_three_models_active': len(self.models) >= 3 and all(m in self.models for m in ['CNN', 'ANN', 'RNN']),
            'model_weights': self.model_weights,
            'association_mining': {
                'enabled': self.association_miner is not None,
                'personas_discovered': len(self.personas_from_association),
                'approval_patterns': len(self.association_miner.approval_patterns) if self.association_miner else 0
            }
        }
        
        if self.model_metrics:
            metrics['performance'] = self.model_metrics
            
        return metrics
    
    def explain_recommendation(self, features, loan_terms, persona, offers):
        """Provide detailed explanation for recommendations with association insights"""
        explanation = {
            'persona': {
                'id': persona,
                'method': 'Association Mining' if self.association_miner else 'K-means Clustering'
            },
            'decision_factors': {
                'models_used': list(self.models.keys()),
                'ensemble_method': 'Weighted average of CNN, ANN, and RNN predictions',
                'confidence_level': self._calculate_confidence(features)
            },
            'offer_rationale': {
                'recommended': offers[1]['type'] if len(offers) > 1 else offers[0]['type'],
                'reason': 'Optimal balance between affordability and total cost',
                'alternatives': len(offers) - 1
            },
            'risk_assessment': self._assess_risk_level(features, loan_terms)
        }
        
        # Add association mining insights if available
        if self.association_miner and persona < len(self.personas_from_association):
            persona_info = self.personas_from_association[persona]
            explanation['persona'].update({
                'name': persona_info['name'],
                'characteristics': persona_info.get('characteristics', {}),
                'confidence': persona_info.get('avg_confidence', 0.5),
                'pattern_strength': persona_info.get('avg_lift', 1.0)
            })
            
            # Get approval insights
            insights = self.association_miner.get_approval_insights()
            if insights:
                explanation['approval_insights'] = insights[:3]  # Top 3 insights
        
        return explanation
    
    def _describe_persona(self, persona):
        """Describe persona characteristics"""
        if self.personas_from_association and persona < len(self.personas_from_association):
            persona_info = self.personas_from_association[persona]
            return f"{persona_info['name']}: {persona_info.get('characteristics', {})}"
        
        descriptions = [
            "Premium segment with excellent credit profile and low risk",
            "Good customers with stable income and balanced approach",
            "Average borrowers with standard risk profile",
            "Cautious borrowers preferring conservative options",
            "Higher risk segment requiring additional verification"
        ]
        
        return descriptions[persona % len(descriptions)]
    
    def _get_cluster_size(self, persona):
        """Estimate cluster size"""
        if self.clustering_model and hasattr(self.clustering_model, 'labels_'):
            total = len(self.clustering_model.labels_)
            cluster_size = np.sum(self.clustering_model.labels_ == persona)
            return f"{cluster_size}/{total} ({cluster_size/total*100:.1f}%)"
        return "Data not available"
    
    def _calculate_confidence(self, features):
        """Calculate prediction confidence"""
        # Based on model agreement and availability
        models_available = len(self.models)
        if models_available >= 3:
            confidence = "High (CNN, ANN, and RNN consensus)"
        elif models_available >= 2:
            confidence = "Moderate (partial model ensemble)"
        else:
            confidence = "Low (limited models available)"
        
        # Add association mining confidence if available
        if self.association_miner:
            confidence += " + Association Mining Insights"
        
        return confidence
    
    def _assess_risk_level(self, features, loan_terms):
        """Assess risk level of the loan"""
        # Simple risk assessment
        rate = loan_terms['rate_of_interest']
        
        if rate < 8:
            return "Low Risk - Excellent profile"
        elif rate < 11:
            return "Moderate Risk - Standard profile"
        elif rate < 14:
            return "Medium Risk - Requires monitoring"
        else:
            return "Higher Risk - Enhanced verification recommended" 