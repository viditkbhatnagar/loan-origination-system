"""
Stable Dynamic Training Script for Loan Origination System
Handles memory issues and provides fallback options
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing TensorFlow with fallback
USE_TENSORFLOW = True
try:
    import tensorflow as tf
    # Configure TensorFlow for Apple Silicon
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU to avoid memory issues
    print("✅ TensorFlow imported successfully (CPU mode)")
except ImportError:
    USE_TENSORFLOW = False
    print("⚠️  TensorFlow not available, using scikit-learn models only")

# Set random seeds
np.random.seed(42)
if USE_TENSORFLOW:
    tf.random.set_seed(42)

class DynamicProcessor:
    """Simplified dynamic data processor"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        
    def prepare_data(self, df, target_col='POLICY_STATUS'):
        """Prepare data dynamically"""
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Handle missing values
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'missing')
        
        # Encode categorical variables
        X = pd.DataFrame()
        
        # Add numeric features
        if numeric_cols:
            X[numeric_cols] = df[numeric_cols]
        
        # Encode categorical features
        for col in categorical_cols:
            if df[col].nunique() <= 10:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col)
                X = pd.concat([X, dummies], axis=1)
            else:
                # Label encode
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(df[col])
        
        # Create interaction features for top numeric columns
        if len(numeric_cols) >= 2:
            top_numeric = sorted(numeric_cols, key=lambda x: df[x].var(), reverse=True)[:3]
            for i in range(len(top_numeric)):
                for j in range(i+1, len(top_numeric)):
                    col1, col2 = top_numeric[i], top_numeric[j]
                    X[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                    X[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        # Prepare target
        y = df[target_col]
        if y.dtype == 'object':
            self.encoders['target'] = LabelEncoder()
            y_encoded = self.encoders['target'].fit_transform(y)
            print(f"Target classes: {dict(zip(self.encoders['target'].classes_, range(len(self.encoders['target'].classes_))))}")
        else:
            y_encoded = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Prepared {X_scaled.shape[1]} features")
        
        # Also prepare loan terms if available
        term_cols = ['rate_of_interest', 'tenure_months', 'sanctioned_amount']
        available_terms = [col for col in term_cols if col in df.columns]
        y_terms = None
        if available_terms:
            y_terms = df[available_terms].values
        
        return X_scaled, y_encoded, y_terms

def build_simple_ann(input_shape, n_classes=2):
    """Build a simple ANN that works well on Apple Silicon"""
    if not USE_TENSORFLOW:
        return None
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(16, activation='relu'),
        
        tf.keras.layers.Dense(1 if n_classes == 2 else n_classes, 
                             activation='sigmoid' if n_classes == 2 else 'softmax')
    ])
    
    return model

def build_simple_cnn(input_shape, n_classes=2):
    """Build a simple 1D CNN for tabular data"""
    if not USE_TENSORFLOW:
        return None
    
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2, padding='same'),
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1 if n_classes == 2 else n_classes,
                             activation='sigmoid' if n_classes == 2 else 'softmax')
    ])
    
    return model

def build_simple_rnn(input_shape, n_classes=2):
    """Build a simple RNN/LSTM for tabular data"""
    if not USE_TENSORFLOW:
        return None
    
    model = tf.keras.Sequential([
        # Reshape for RNN: (batch, timesteps=1, features)
        tf.keras.layers.Reshape((1, input_shape), input_shape=(input_shape,)),
        
        # LSTM layers
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        
        # Dense layers
        tf.keras.layers.Dense(16, activation='relu'),
        
        tf.keras.layers.Dense(1 if n_classes == 2 else n_classes,
                             activation='sigmoid' if n_classes == 2 else 'softmax')
    ])
    
    return model

def train_stable_dynamic_models():
    """Train models with stability and fallback options"""
    
    print("🚀 Stable Dynamic Loan Origination Training")
    print("=" * 60)
    
    # Load dataset
    data_file = 'data/Insurance_Enhanced.csv'
    if not os.path.exists(data_file):
        print("❌ Dataset not found!")
        return False
    
    df = pd.read_csv(data_file)
    print(f"📊 Dataset loaded: {df.shape}")
    
    # Initialize processor
    processor = DynamicProcessor()
    
    # Prepare data
    X, y, y_terms = processor.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if y_terms is not None:
        _, _, y_terms_train, y_terms_test = train_test_split(
            X, y_terms, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    
    # Create models directory
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Store results
    results = {}
    
    # Train scikit-learn models (always available)
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    results['RandomForest'] = rf_metrics
    print(f"✅ Random Forest Accuracy: {rf_metrics['accuracy']:.4f}")
    
    # Save Random Forest
    joblib.dump(rf_model, 'models/saved_models/rf_model.pkl')
    
    # Train Gradient Boosting
    print("\n🚀 Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    
    gb_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb, average='weighted'),
        'recall': recall_score(y_test, y_pred_gb, average='weighted'),
        'f1': f1_score(y_test, y_pred_gb, average='weighted')
    }
    results['GradientBoosting'] = gb_metrics
    print(f"✅ Gradient Boosting Accuracy: {gb_metrics['accuracy']:.4f}")
    
    # Save Gradient Boosting
    joblib.dump(gb_model, 'models/saved_models/gb_model.pkl')
    
    # Train deep learning models if TensorFlow is available
    if USE_TENSORFLOW:
        print("\n🧠 Training Neural Networks...")
        
        # Common callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        
        # Build and train ANN
        print("\n  Building ANN...")
        ann_model = build_simple_ann(X_train.shape[1])
        if ann_model:
            ann_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            history = ann_model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=30,
                batch_size=32,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Evaluate
            ann_loss, ann_acc = ann_model.evaluate(X_test, y_test, verbose=0)
            results['ANN'] = {'accuracy': ann_acc, 'loss': ann_loss}
            print(f"  ✅ ANN Accuracy: {ann_acc:.4f}")
            
            # Save ANN
            ann_model.save('models/saved_models/ann_model.h5')
        
        # Build and train CNN
        print("\n  Building CNN...")
        cnn_model = build_simple_cnn(X_train.shape[1])
        if cnn_model:
            cnn_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            history_cnn = cnn_model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=30,
                batch_size=32,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Evaluate
            cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
            results['CNN'] = {'accuracy': cnn_acc, 'loss': cnn_loss}
            print(f"  ✅ CNN Accuracy: {cnn_acc:.4f}")
            
            # Save CNN
            cnn_model.save('models/saved_models/cnn_model.h5')
        
        # Build and train RNN
        print("\n  Building RNN...")
        rnn_model = build_simple_rnn(X_train.shape[1])
        if rnn_model:
            rnn_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            history_rnn = rnn_model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=30,
                batch_size=32,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Evaluate
            rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test, verbose=0)
            results['RNN'] = {'accuracy': rnn_acc, 'loss': rnn_loss}
            print(f"  ✅ RNN Accuracy: {rnn_acc:.4f}")
            
            # Save RNN
            rnn_model.save('models/saved_models/rnn_model.h5')
    
    # Train clustering model
    print("\n👥 Training Clustering Model...")
    
    # Filter to positive class for clustering
    positive_mask = y_train == 1
    X_cluster = X_train[positive_mask] if positive_mask.any() else X_train
    
    # Find optimal clusters
    best_k = 5
    best_score = -1
    
    for k in range(3, min(8, len(X_cluster) // 10)):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_temp.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        print(f"  k={k}: Silhouette Score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    
    # Train final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    print(f"✅ Optimal clusters: {best_k}")
    
    # Save clustering model
    joblib.dump(kmeans, 'models/saved_models/kmeans_model.pkl')
    
    # Train loan terms predictor if data available
    if y_terms is not None:
        print("\n💰 Training Loan Terms Predictor...")
        terms_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Train only on positive class
        positive_train = y_train == 1
        if positive_train.any():
            terms_model.fit(X_train[positive_train], y_terms_train[positive_train])
            
            # Evaluate
            positive_test = y_test == 1
            if positive_test.any():
                y_pred_terms = terms_model.predict(X_test[positive_test])
                mae_rate = np.mean(np.abs(y_pred_terms[:, 0] - y_terms_test[positive_test][:, 0]))
                print(f"✅ MAE Interest Rate: {mae_rate:.4f}")
            
            # Save terms model
            joblib.dump(terms_model, 'models/saved_models/terms_model.pkl')
    
    # Save preprocessor and results
    joblib.dump(processor, 'models/saved_models/data_processor.pkl')
    joblib.dump(processor.scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(processor.encoders, 'models/saved_models/label_encoders.pkl')
    joblib.dump(processor.feature_names, 'models/saved_models/feature_names.pkl')
    
    # Save results
    with open('models/saved_models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save simplified metrics for compatibility
    best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
    model_metrics = {
        'accuracy': best_model[1].get('accuracy', 0),
        'precision': best_model[1].get('precision', 0),
        'recall': best_model[1].get('recall', 0),
        'f1': best_model[1].get('f1', 0),
        'n_clusters': best_k,
        'silhouette_score': best_score
    }
    joblib.dump(model_metrics, 'models/saved_models/model_metrics.pkl')
    
    # Create visualization
    create_simple_visualization(results)
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📊 Model Performance Summary:")
    for model_name, metrics in results.items():
        print(f"   {model_name}: {metrics.get('accuracy', 0):.4f} accuracy")
    
    print(f"\n📁 Models saved in: models/saved_models/")
    print("🚀 Ready to run the application!")
    
    return True

def create_simple_visualization(results):
    """Create a simple visualization of results"""
    plots_dir = 'static/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Model comparison
    plt.figure(figsize=(10, 6))
    
    models = list(results.keys())
    accuracies = [results[m].get('accuracy', 0) for m in models]
    
    plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'][:len(models)])
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/model_comparison.png', dpi=150)
    plt.close()
    
    print(f"✅ Visualization saved to {plots_dir}/")

if __name__ == "__main__":
    train_stable_dynamic_models() 