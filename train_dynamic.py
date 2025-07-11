"""
Dynamic Loan Origination System Training Script
Trains multiple deep learning models dynamically with automatic data processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, 
                                     Dropout, BatchNormalization, Input, Embedding,
                                     Bidirectional, GlobalAveragePooling1D, concatenate)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DynamicDataProcessor:
    """Dynamically process any loan dataset"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.categorical_mappings = {}
        
    def analyze_dataset(self, df):
        """Dynamically analyze dataset structure"""
        print("\n📊 Dataset Analysis:")
        print(f"   Shape: {df.shape}")
        print(f"   Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        print(f"   Numeric columns: {len(numeric_cols)}")
        print(f"   Categorical columns: {len(categorical_cols)}")
        print(f"   Datetime columns: {len(datetime_cols)}")
        
        # Missing values analysis
        missing = df.isnull().sum()
        if missing.any():
            print("\n   Missing values:")
            for col, count in missing[missing > 0].items():
                print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
        
        return numeric_cols, categorical_cols, datetime_cols
    
    def create_dynamic_features(self, df):
        """Create features dynamically based on data patterns"""
        features = df.copy()
        
        # Dynamic feature engineering
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        # Create ratios and interactions for numeric features
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if features[col2].nunique() > 1 and not (features[col2] == 0).all():
                    # Ratio features
                    features[f'{col1}_to_{col2}_ratio'] = features[col1] / (features[col2] + 1e-8)
                    
                    # Interaction features
                    features[f'{col1}_times_{col2}'] = features[col1] * features[col2]
        
        # Create statistical features
        for col in numeric_cols:
            if features[col].nunique() > 10:
                # Binning
                features[f'{col}_quartile'] = pd.qcut(features[col], q=4, labels=False, duplicates='drop')
                
                # Distance from mean
                mean_val = features[col].mean()
                std_val = features[col].std()
                features[f'{col}_z_score'] = (features[col] - mean_val) / (std_val + 1e-8)
        
        # Time-based features if datetime columns exist
        datetime_cols = features.select_dtypes(include=['datetime']).columns
        for col in datetime_cols:
            features[f'{col}_year'] = features[col].dt.year
            features[f'{col}_month'] = features[col].dt.month
            features[f'{col}_day'] = features[col].dt.day
            features[f'{col}_dayofweek'] = features[col].dt.dayofweek
            features[f'{col}_quarter'] = features[col].dt.quarter
        
        print(f"✅ Created {len(features.columns) - len(df.columns)} new features dynamically")
        return features
    
    def prepare_data(self, df, target_column=None):
        """Dynamically prepare data for training"""
        
        # If no target specified, try to identify it
        if target_column is None:
            potential_targets = ['POLICY_STATUS', 'loan_status', 'approved', 'status', 'target']
            for col in potential_targets:
                if col in df.columns:
                    target_column = col
                    break
        
        if target_column is None:
            raise ValueError("Could not identify target column. Please specify it.")
        
        print(f"\n🎯 Target column: {target_column}")
        
        # Create dynamic features
        df_enhanced = self.create_dynamic_features(df)
        
        # Separate features and target
        y = df_enhanced[target_column]
        X = df_enhanced.drop(columns=[target_column])
        
        # Handle categorical target
        if y.dtype == 'object':
            self.encoders['target'] = LabelEncoder()
            y_encoded = self.encoders['target'].fit_transform(y)
            print(f"   Target classes: {dict(zip(self.encoders['target'].classes_, range(len(self.encoders['target'].classes_))))}")
        else:
            y_encoded = y.values
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].nunique() < 20:  # One-hot encode if few categories
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
                X.drop(columns=[col], inplace=True)
            else:  # Label encode if many categories
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].fillna('missing'))
        
        # Handle missing values in numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Prepared {X_scaled.shape[1]} features from {df.shape[1]} original columns")
        
        return X_scaled, y_encoded, self.feature_names

class DynamicModelBuilder:
    """Build various deep learning models dynamically"""
    
    def __init__(self, input_shape, n_classes=2, task_type='classification'):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.task_type = task_type
        
    def build_dynamic_cnn(self, name="CNN"):
        """Build a dynamic CNN for tabular data"""
        model = Sequential(name=name)
        
        # Reshape for CNN (add channel dimension)
        model.add(tf.keras.layers.Reshape((self.input_shape, 1), input_shape=(self.input_shape,)))
        
        # Dynamic CNN layers based on input size
        n_filters = min(128, max(32, self.input_shape // 4))
        
        # First conv block
        model.add(Conv1D(n_filters, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2, padding='same'))
        model.add(Dropout(0.3))
        
        # Second conv block
        model.add(Conv1D(n_filters * 2, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2, padding='same'))
        model.add(Dropout(0.3))
        
        # Third conv block if input is large
        if self.input_shape > 50:
            model.add(Conv1D(n_filters * 4, 3, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(GlobalAveragePooling1D())
        else:
            model.add(Flatten())
        
        # Dense layers
        model.add(Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        
        # Output layer
        if self.task_type == 'classification':
            if self.n_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(self.n_classes, activation='softmax'))
        else:
            model.add(Dense(3))  # For regression (rate, tenure, amount)
        
        return model
    
    def build_dynamic_ann(self, name="ANN"):
        """Build a dynamic ANN with automatic architecture"""
        model = Sequential(name=name)
        
        # Dynamic layer sizes based on input
        layer_sizes = []
        current_size = self.input_shape
        
        # Create pyramid architecture
        while current_size > 32:
            next_size = int(current_size * 0.6)
            layer_sizes.append(max(32, next_size))
            current_size = next_size
        
        # Input layer
        model.add(Dense(layer_sizes[0], activation='relu', input_shape=(self.input_shape,),
                       kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        for size in layer_sizes[1:]:
            model.add(Dense(size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        
        # Additional dense layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        
        # Output layer
        if self.task_type == 'classification':
            if self.n_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(self.n_classes, activation='softmax'))
        else:
            model.add(Dense(3))
        
        return model
    
    def build_dynamic_rnn(self, name="RNN"):
        """Build a dynamic RNN/LSTM model"""
        model = Sequential(name=name)
        
        # Reshape for RNN
        model.add(tf.keras.layers.Reshape((1, self.input_shape), input_shape=(self.input_shape,)))
        
        # Dynamic LSTM layers
        lstm_units = min(256, max(64, self.input_shape * 2))
        
        # Bidirectional LSTM layers
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        model.add(Dropout(0.3))
        
        model.add(Bidirectional(LSTM(lstm_units // 2, return_sequences=True)))
        model.add(Dropout(0.3))
        
        model.add(Bidirectional(LSTM(lstm_units // 4)))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer
        if self.task_type == 'classification':
            if self.n_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(self.n_classes, activation='softmax'))
        else:
            model.add(Dense(3))
        
        return model
    
    def build_ensemble_model(self):
        """Build an ensemble model combining CNN, ANN, and RNN"""
        input_layer = Input(shape=(self.input_shape,))
        
        # CNN branch
        cnn_reshape = tf.keras.layers.Reshape((self.input_shape, 1))(input_layer)
        cnn = Conv1D(64, 3, activation='relu', padding='same')(cnn_reshape)
        cnn = GlobalAveragePooling1D()(cnn)
        cnn = Dense(64, activation='relu')(cnn)
        
        # ANN branch
        ann = Dense(128, activation='relu')(input_layer)
        ann = BatchNormalization()(ann)
        ann = Dropout(0.3)(ann)
        ann = Dense(64, activation='relu')(ann)
        
        # RNN branch
        rnn_reshape = tf.keras.layers.Reshape((1, self.input_shape))(input_layer)
        rnn = LSTM(64)(rnn_reshape)
        
        # Combine branches
        combined = concatenate([cnn, ann, rnn])
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(64, activation='relu')(combined)
        
        # Output
        if self.task_type == 'classification':
            if self.n_classes == 2:
                output = Dense(1, activation='sigmoid')(combined)
            else:
                output = Dense(self.n_classes, activation='softmax')(combined)
        else:
            output = Dense(3)(combined)
        
        model = Model(inputs=input_layer, outputs=output, name="Ensemble")
        return model

def train_dynamic_models():
    """Main training function with dynamic data handling"""
    
    print("🚀 Dynamic Loan Origination System Training")
    print("=" * 60)
    
    # Load dataset dynamically
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    if not data_files:
        print("❌ No CSV files found in data directory!")
        return False
    
    # Use the first CSV file found
    data_file = os.path.join('data', data_files[0])
    print(f"📁 Loading dataset: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Initialize dynamic processor
    processor = DynamicDataProcessor()
    
    # Analyze dataset
    numeric_cols, categorical_cols, datetime_cols = processor.analyze_dataset(df)
    
    # Prepare data dynamically
    try:
        X, y, feature_names = processor.prepare_data(df)
    except ValueError as e:
        # If target not found, list columns for user
        print(f"\n❌ {e}")
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"   {i}: {col}")
        return False
    
    # Determine task type
    n_classes = len(np.unique(y))
    task_type = 'classification' if n_classes < 10 else 'regression'
    print(f"\n🎯 Task type: {task_type} ({n_classes} classes)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if task_type == 'classification' else None
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {X_train.shape}")
    print(f"   Testing: {X_test.shape}")
    
    # Initialize model builder
    builder = DynamicModelBuilder(X_train.shape[1], n_classes, task_type)
    
    # Create models directory
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Models to train
    models = {
        'CNN': builder.build_dynamic_cnn(),
        'ANN': builder.build_dynamic_ann(),
        'RNN': builder.build_dynamic_rnn(),
        'Ensemble': builder.build_ensemble_model()
    }
    
    # Compile models
    for name, model in models.items():
        if task_type == 'classification':
            if n_classes == 2:
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                )
            else:
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
        else:
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        print(f"\n🏗️ {name} Architecture:")
        print(f"   Total parameters: {model.count_params():,}")
    
    # Training callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, monitor='val_loss')
    ]
    
    # Store results
    results = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n🧠 Training {name}...")
        print("-" * 50)
        
        # Create model checkpoint
        checkpoint = ModelCheckpoint(
            f'models/saved_models/{name.lower()}_best.h5',
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks + [checkpoint],
            verbose=1
        )
        
        # Evaluate model
        if task_type == 'classification':
            y_pred_proba = model.predict(X_test)
            if n_classes == 2:
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'history': history.history
            }
            
            print(f"\n📊 {name} Performance:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
        else:
            mae = model.evaluate(X_test, y_test, verbose=0)[1]
            results[name] = {
                'mae': mae,
                'history': history.history
            }
            print(f"\n📊 {name} Performance:")
            print(f"   MAE: {mae:.4f}")
        
        # Save model
        model.save(f'models/saved_models/{name.lower()}_model.h5')
    
    # Train additional models (Random Forest, KMeans)
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f"   Accuracy: {rf_accuracy:.4f}")
    
    # Dynamic clustering
    print("\n👥 Training Dynamic Clustering...")
    
    # Find optimal clusters using elbow method
    if task_type == 'classification' and n_classes == 2:
        # Only cluster positive class
        positive_mask = y_train == 1
        X_cluster = X_train[positive_mask]
    else:
        X_cluster = X_train
    
    # Try different cluster numbers
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(10, len(X_cluster) // 100))
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_cluster)
        inertias.append(kmeans_temp.inertia_)
        
        if k < len(X_cluster):
            from sklearn.metrics import silhouette_score
            score = silhouette_score(X_cluster, kmeans_temp.labels_)
            silhouette_scores.append(score)
            print(f"   k={k}: Silhouette Score = {score:.4f}")
    
    # Select best k
    if silhouette_scores:
        best_k = K_range[np.argmax(silhouette_scores)]
    else:
        best_k = 5
    
    print(f"\n✅ Optimal clusters: {best_k}")
    
    # Train final clustering model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    
    # Save all models and processors
    print("\n💾 Saving models and processors...")
    
    # Save sklearn models
    joblib.dump(rf_model, 'models/saved_models/rf_model.pkl')
    joblib.dump(kmeans, 'models/saved_models/kmeans_model.pkl')
    
    # Save processors
    joblib.dump(processor, 'models/saved_models/data_processor.pkl')
    joblib.dump(feature_names, 'models/saved_models/feature_names.pkl')
    
    # Save results
    with open('models/saved_models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    create_dynamic_visualizations(results, X_test, y_test, feature_names)
    
    print("\n" + "=" * 60)
    print("🎉 DYNAMIC TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📊 Best performing models:")
    
    if task_type == 'classification':
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"   {best_model[0]}: {best_model[1]['accuracy']:.4f} accuracy")
    else:
        best_model = min(results.items(), key=lambda x: x[1]['mae'])
        print(f"   {best_model[0]}: {best_model[1]['mae']:.4f} MAE")
    
    print(f"\n📁 Models saved in: models/saved_models/")
    print("🚀 Ready to run the application!")
    
    return True

def create_dynamic_visualizations(results, X_test, y_test, feature_names):
    """Create comprehensive visualizations"""
    
    plots_dir = 'static/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Model comparison plot
    plt.figure(figsize=(15, 10))
    
    # Performance comparison
    plt.subplot(2, 2, 1)
    models = list(results.keys())
    if 'accuracy' in results[models[0]]:
        accuracies = [results[m]['accuracy'] for m in models]
        plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Training history
    plt.subplot(2, 2, 2)
    for model_name, result in results.items():
        if 'history' in result and 'loss' in result['history']:
            plt.plot(result['history']['loss'], label=f'{model_name} Train')
            if 'val_loss' in result['history']:
                plt.plot(result['history']['val_loss'], '--', label=f'{model_name} Val')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Feature importance visualization using PCA
    plt.subplot(2, 2, 3)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test[:500])  # Sample for visualization
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test[:500], cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title('Data Distribution (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    # Model metrics comparison
    plt.subplot(2, 2, 4)
    if 'accuracy' in results[models[0]]:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[m][metric] for m in models]
            plt.bar(x + i*width, values, width, label=metric.capitalize())
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Detailed Metrics Comparison')
        plt.xticks(x + width*1.5, models)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/dynamic_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {plots_dir}/")

if __name__ == "__main__":
    train_dynamic_models() 