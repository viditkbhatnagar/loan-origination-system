# simple_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def train_simple_models():
    """Train models without TensorFlow"""
    print("Loading dataset...")
    df = pd.read_csv('data/Insurance_Enhanced.csv')
    print(f"Dataset loaded: {len(df)} rows")
    
    # Prepare features
    categorical_cols = ['PI_GENDER', 'PI_OCCUPATION', 'ZONE', 'PAYMENT_MODE', 
                       'EARLY_NON', 'MEDICAL_NONMED', 'PI_STATE']
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Select features
    feature_cols = ['PI_AGE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED'] + [f'{col}_encoded' for col in categorical_cols]
    X = df[feature_cols].values
    
    # Create eligibility target
    y_eligibility = (df['POLICY_STATUS'] == 'Approved Death Claim').astype(int)
    y_terms = df[['rate_of_interest', 'tenure_months', 'sanctioned_amount']].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_elig_train, y_elig_test, y_terms_train, y_terms_test = train_test_split(
        X_scaled, y_eligibility, y_terms, test_size=0.2, random_state=42
    )
    
    # Create models directory
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Train Random Forest for eligibility
    print("Training eligibility model...")
    rf_eligibility = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_eligibility.fit(X_train, y_elig_train)
    
    # Evaluate
    y_pred = rf_eligibility.predict(X_test)
    accuracy = accuracy_score(y_elig_test, y_pred)
    print(f"Eligibility model accuracy: {accuracy:.4f}")
    
    # Train Random Forest for terms (only eligible customers)
    print("Training terms prediction model...")
    eligible_mask = y_elig_train == 1
    X_train_eligible = X_train[eligible_mask]
    y_terms_train_eligible = y_terms_train[eligible_mask]
    
    rf_terms = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_terms.fit(X_train_eligible, y_terms_train_eligible)
    
    # Train KMeans for clustering
    print("Training clustering model...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_train_eligible)
    
    # Save models
    joblib.dump(rf_eligibility, 'models/saved_models/eligibility_model.pkl')
    joblib.dump(rf_terms, 'models/saved_models/terms_model.pkl')
    joblib.dump(kmeans, 'models/saved_models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(encoders, 'models/saved_models/label_encoders.pkl')
    
    print("Models trained and saved successfully!")
    return True

if __name__ == "__main__":
    train_simple_models()