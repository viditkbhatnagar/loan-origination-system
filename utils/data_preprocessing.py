"""
Dynamic Data Preprocessing Module
Handles any loan dataset structure automatically
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import os

class DataPreprocessor:
    """Dynamic data preprocessor that adapts to any dataset"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_selector = None
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.text_features = []
        self.is_fitted = False
        
    def analyze_column_types(self, df):
        """Dynamically identify column types"""
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                self.numeric_features.append(col)
            elif df[col].dtype == 'object':
                # Check if it might be datetime
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    if pd.to_datetime(df[col], errors='coerce').notna().sum() > len(df) * 0.5:
                        self.datetime_features.append(col)
                    else:
                        # Check if it's text (high cardinality) or categorical
                        if df[col].nunique() > 50 or df[col].str.len().mean() > 50:
                            self.text_features.append(col)
                        else:
                            self.categorical_features.append(col)
                except:
                    self.categorical_features.append(col)
            elif df[col].dtype == 'datetime64[ns]':
                self.datetime_features.append(col)
        
        print(f"📊 Column Analysis:")
        print(f"   Numeric: {len(self.numeric_features)}")
        print(f"   Categorical: {len(self.categorical_features)}")
        print(f"   Datetime: {len(self.datetime_features)}")
        print(f"   Text: {len(self.text_features)}")
    
    def create_datetime_features(self, df):
        """Extract features from datetime columns"""
        for col in self.datetime_features:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_quarter'] = df[col].dt.quarter
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Days since a reference date
                ref_date = df[col].min()
                df[f'{col}_days_since_start'] = (df[col] - ref_date).dt.days
                
                # Drop original datetime column
                df = df.drop(columns=[col])
        
        return df
    
    def create_text_features(self, df):
        """Extract features from text columns"""
        for col in self.text_features:
            if col in df.columns:
                # Basic text features
                df[f'{col}_length'] = df[col].fillna('').str.len()
                df[f'{col}_word_count'] = df[col].fillna('').str.split().str.len()
                
                # Drop original text column for now (could use TF-IDF later)
                df = df.drop(columns=[col])
        
        return df
    
    def handle_missing_values(self, df):
        """Intelligent missing value handling"""
        # Numeric columns
        for col in self.numeric_features:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df)
                
                if missing_pct > 0:
                    if missing_pct < 0.05:  # Low missing - use mean
                        self.imputers[col] = SimpleImputer(strategy='mean')
                    elif missing_pct < 0.15:  # Moderate missing - use median
                        self.imputers[col] = SimpleImputer(strategy='median')
                    else:  # High missing - use KNN
                        self.imputers[col] = KNNImputer(n_neighbors=5)
                    
                    df[col] = self.imputers[col].fit_transform(df[[col]])
        
        # Categorical columns
        for col in self.categorical_features:
            if col in df.columns:
                # Fill with mode or 'missing' indicator
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'missing'
                    df[col] = df[col].fillna(mode_val)
        
        return df
    
    def create_interaction_features(self, df, max_features=20):
        """Create interaction features between numeric columns"""
        numeric_cols = [col for col in self.numeric_features if col in df.columns]
        
        if len(numeric_cols) >= 2:
            # Select top numeric features based on variance
            variances = df[numeric_cols].var()
            top_cols = variances.nlargest(min(5, len(numeric_cols))).index.tolist()
            
            # Create interactions
            for i, col1 in enumerate(top_cols):
                for col2 in top_cols[i+1:]:
                    # Multiplication
                    df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                    
                    # Division (avoid division by zero)
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    
                    # Difference
                    df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                    
                    if len(df.columns) >= max_features:
                        break
                if len(df.columns) >= max_features:
                    break
        
        return df
    
    def encode_categorical_features(self, df):
        """Smart categorical encoding"""
        for col in self.categorical_features:
            if col in df.columns:
                n_unique = df[col].nunique()
                
                if n_unique == 2:  # Binary
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                elif n_unique <= 10:  # Low cardinality - one-hot encode
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                else:  # High cardinality - target encoding or label encoding
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def create_statistical_features(self, df):
        """Create statistical features for numeric columns"""
        numeric_cols = [col for col in self.numeric_features if col in df.columns]
        
        if len(numeric_cols) > 0:
            # Row-wise statistics
            df['numeric_mean'] = df[numeric_cols].mean(axis=1)
            df['numeric_std'] = df[numeric_cols].std(axis=1)
            df['numeric_min'] = df[numeric_cols].min(axis=1)
            df['numeric_max'] = df[numeric_cols].max(axis=1)
            df['numeric_range'] = df['numeric_max'] - df['numeric_min']
            
            # Z-scores for each numeric column
            for col in numeric_cols[:10]:  # Limit to avoid too many features
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
        
        return df
    
    def prepare_training_data(self, df, target_column=None):
        """Prepare data for model training"""
        # Reset state
        self.__init__()
        
        # Analyze column types
        self.analyze_column_types(df)
        
        # Find target column if not specified
        if target_column is None:
            potential_targets = ['POLICY_STATUS', 'loan_status', 'status', 'approved', 
                               'target', 'label', 'y', 'outcome']
            for col in potential_targets:
                if col in df.columns:
                    target_column = col
                    break
        
        # Separate features and target
        if target_column and target_column in df.columns:
            y = df[target_column].copy()
            X = df.drop(columns=[target_column]).copy()
            
            # Encode target if categorical
            if y.dtype == 'object':
                self.label_encoders['target'] = LabelEncoder()
                y_encoded = self.label_encoders['target'].fit_transform(y)
            else:
                y_encoded = y.values
        else:
            X = df.copy()
            y_encoded = None
        
        # Feature engineering pipeline
        print("\n🔧 Feature Engineering Pipeline:")
        
        # Handle datetime features
        X = self.create_datetime_features(X)
        print("   ✓ Datetime features created")
        
        # Handle text features
        X = self.create_text_features(X)
        print("   ✓ Text features processed")
        
        # Handle missing values
        X = self.handle_missing_values(X)
        print("   ✓ Missing values handled")
        
        # Create statistical features
        X = self.create_statistical_features(X)
        print("   ✓ Statistical features created")
        
        # Create interaction features
        X = self.create_interaction_features(X)
        print("   ✓ Interaction features created")
        
        # Encode categorical features
        X = self.encode_categorical_features(X)
        print("   ✓ Categorical features encoded")
        
        # Update feature lists after transformations
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        print(f"   ✓ Features scaled to shape {X_scaled.shape}")
        
        # Create additional target variables for loan terms
        y_terms = None
        if target_column and y_encoded is not None:
            # Check if we have loan term columns
            term_columns = ['rate_of_interest', 'tenure_months', 'sanctioned_amount']
            available_terms = [col for col in term_columns if col in df.columns]
            
            if available_terms:
                y_terms = df[available_terms].values
                print(f"   ✓ Found {len(available_terms)} loan term targets")
        
        self.is_fitted = True
        
        if y_terms is not None:
            return X_scaled, y_encoded, y_terms
        elif y_encoded is not None:
            return X_scaled, y_encoded
        else:
            return X_scaled
    
    def preprocess_input(self, input_data):
        """Preprocess new input data using fitted transformers"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first using prepare_training_data()")
        
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Apply same transformations
        df = self.create_datetime_features(df)
        df = self.create_text_features(df)
        
        # Handle missing values using fitted imputers
        for col, imputer in self.imputers.items():
            if col in df.columns:
                df[col] = imputer.transform(df[[col]])
        
        # Fill any remaining missing values
        df = df.fillna(0)
        
        # Create statistical features
        df = self.create_statistical_features(df)
        df = self.create_interaction_features(df)
        
        # Encode categorical features using fitted encoders
        for col, encoder in self.label_encoders.items():
            if col in df.columns and col != 'target':
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select and order features
        df = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def save(self, path='models/saved_models/'):
        """Save preprocessor state"""
        os.makedirs(path, exist_ok=True)
        
        # Save main preprocessor
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputers': self.imputers,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'datetime_features': self.datetime_features,
            'text_features': self.text_features,
            'is_fitted': self.is_fitted
        }, os.path.join(path, 'preprocessor_state.pkl'))
        
        print(f"✅ Preprocessor saved to {path}")
    
    def load(self, path='models/saved_models/'):
        """Load preprocessor state"""
        state_file = os.path.join(path, 'preprocessor_state.pkl')
        
        if os.path.exists(state_file):
            state = joblib.load(state_file)
            self.scaler = state['scaler']
            self.label_encoders = state['label_encoders']
            self.imputers = state['imputers']
            self.feature_names = state['feature_names']
            self.numeric_features = state['numeric_features']
            self.categorical_features = state['categorical_features']
            self.datetime_features = state['datetime_features']
            self.text_features = state['text_features']
            self.is_fitted = state['is_fitted']
            print(f"✅ Preprocessor loaded from {path}")
        else:
            print(f"⚠️  No preprocessor state found at {path}")