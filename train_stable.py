import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)

def create_plots_directory():
    """Create plots directory"""
    plots_dir = 'static/plots'
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def prepare_data(df):
    """Prepare data for training"""
    # Encode categorical variables
    categorical_cols = ['PI_GENDER', 'PI_OCCUPATION', 'ZONE', 'PAYMENT_MODE', 
                       'EARLY_NON', 'MEDICAL_NONMED', 'PI_STATE']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Select features
    feature_cols = ['PI_AGE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED'] + [f'{col}_encoded' for col in categorical_cols]
    X = df[feature_cols].values
    
    # Create targets
    y_eligibility = (df['POLICY_STATUS'] == 'Approved Death Claim').astype(int)
    y_terms = df[['rate_of_interest', 'tenure_months', 'sanctioned_amount']].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_eligibility, y_terms, scaler, encoders

def create_visualizations(df, X_scaled, y_eligibility, cluster_labels, model_metrics, plots_dir):
    """Create comprehensive visualizations"""
    
    # Data analysis plot
    plt.figure(figsize=(20, 12))
    
    # Age distribution
    plt.subplot(3, 4, 1)
    approved = df[df['POLICY_STATUS'] == 'Approved Death Claim']['PI_AGE']
    rejected = df[df['POLICY_STATUS'] == 'Repudiate Death']['PI_AGE']
    plt.hist([approved, rejected], bins=20, alpha=0.7, label=['Approved', 'Rejected'], color=['green', 'red'])
    plt.title('Age Distribution by Loan Status')
    plt.xlabel('Age')
    plt.legend()
    
    # Income distribution
    plt.subplot(3, 4, 2)
    approved_income = df[df['POLICY_STATUS'] == 'Approved Death Claim']['PI_ANNUAL_INCOME']
    rejected_income = df[df['POLICY_STATUS'] == 'Repudiate Death']['PI_ANNUAL_INCOME']
    plt.hist([approved_income, rejected_income], bins=20, alpha=0.7, label=['Approved', 'Rejected'], color=['green', 'red'])
    plt.title('Income Distribution by Loan Status')
    plt.xlabel('Annual Income')
    plt.legend()
    
    # Gender distribution
    plt.subplot(3, 4, 3)
    gender_status = pd.crosstab(df['PI_GENDER'], df['POLICY_STATUS'])
    gender_status.plot(kind='bar', ax=plt.gca(), color=['red', 'green'])
    plt.title('Gender vs Loan Status')
    plt.xticks(rotation=0)
    
    # Rate of interest distribution
    plt.subplot(3, 4, 4)
    plt.hist(df['rate_of_interest'], bins=20, alpha=0.7, color='skyblue')
    plt.title('Interest Rate Distribution')
    plt.xlabel('Interest Rate (%)')
    
    # Model performance
    plt.subplot(3, 4, 5)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [model_metrics['accuracy'], model_metrics['precision'], 
              model_metrics['recall'], model_metrics['f1']]
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.title('Model Performance')
    plt.ylim(0, 1)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', ha='center')
    
    # Feature importance
    plt.subplot(3, 4, 6)
    feature_names = ['Age', 'Income', 'Sum Assured', 'Gender', 'Occupation', 'Zone', 'Payment', 'Early', 'Medical', 'State']
    importance = model_metrics.get('feature_importance', [0.1] * len(feature_names))[:len(feature_names)]
    plt.barh(feature_names, importance)
    plt.title('Feature Importance')
    
    # Cluster visualization
    plt.subplot(3, 4, 7)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled[y_eligibility == 1])
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Customer Clusters (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    # Cluster distribution
    plt.subplot(3, 4, 8)
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    plt.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Customer Distribution')
    
    # Additional plots
    plt.subplot(3, 4, 9)
    plt.scatter(df['PI_AGE'], df['PI_ANNUAL_INCOME'], 
                c=df['POLICY_STATUS'].map({'Approved Death Claim': 1, 'Repudiate Death': 0}),
                alpha=0.6, cmap='RdYlGn')
    plt.xlabel('Age')
    plt.ylabel('Annual Income')
    plt.title('Age vs Income (Colored by Status)')
    plt.colorbar(label='Approved')
    
    plt.subplot(3, 4, 10)
    zone_counts = df['ZONE'].value_counts()
    plt.pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%')
    plt.title('Applications by Zone')
    
    plt.subplot(3, 4, 11)
    occupation_counts = df['PI_OCCUPATION'].value_counts().head(8)
    plt.barh(occupation_counts.index, occupation_counts.values)
    plt.title('Top Occupations')
    
    plt.subplot(3, 4, 12)
    plt.hist(df['sanctioned_amount'], bins=20, alpha=0.7, color='gold')
    plt.title('Sanctioned Amount Distribution')
    plt.xlabel('Amount (₹)')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {plots_dir}/comprehensive_analysis.png")

def train_stable_models():
    """Train models with stable configuration"""
    
    print("🚀 Starting Stable Model Training")
    print("=" * 50)
    
    # Create directories
    os.makedirs('models/saved_models', exist_ok=True)
    plots_dir = create_plots_directory()
    
    # Load data
    print("📊 Loading dataset...")
    df = pd.read_csv('data/Insurance_Enhanced.csv')
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Prepare data
    print("🔧 Preparing data...")
    X, y_eligibility, y_terms, scaler, encoders = prepare_data(df)
    
    # Split data
    X_train, X_test, y_elig_train, y_elig_test, y_terms_train, y_terms_test = train_test_split(
        X, y_eligibility, y_terms, test_size=0.2, random_state=42, stratify=y_eligibility
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train eligibility model
    print("\n🧠 Training Eligibility Model...")
    rf_eligibility = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_eligibility.fit(X_train, y_elig_train)
    
    # Evaluate eligibility model
    y_pred = rf_eligibility.predict(X_test)
    y_pred_proba = rf_eligibility.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_elig_test, y_pred)
    precision = precision_score(y_elig_test, y_pred)
    recall = recall_score(y_elig_test, y_pred)
    f1 = f1_score(y_elig_test, y_pred)
    
    print(f"✅ Eligibility Model Performance:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Train terms model (only on eligible customers)
    print("\n💰 Training Terms Prediction Model...")
    eligible_mask = y_elig_train == 1
    X_train_eligible = X_train[eligible_mask]
    y_terms_train_eligible = y_terms_train[eligible_mask]
    
    rf_terms = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_terms.fit(X_train_eligible, y_terms_train_eligible)
    
    # Evaluate terms model
    eligible_mask_test = y_elig_test == 1
    if np.sum(eligible_mask_test) > 0:
        X_test_eligible = X_test[eligible_mask_test]
        y_terms_test_eligible = y_terms_test[eligible_mask_test]
        y_pred_terms = rf_terms.predict(X_test_eligible)
        
        mae_rate = np.mean(np.abs(y_pred_terms[:, 0] - y_terms_test_eligible[:, 0]))
        mae_tenure = np.mean(np.abs(y_pred_terms[:, 1] - y_terms_test_eligible[:, 1]))
        mae_amount = np.mean(np.abs(y_pred_terms[:, 2] - y_terms_test_eligible[:, 2]))
        
        print(f"✅ Terms Model Performance:")
        print(f"   MAE Interest Rate: {mae_rate:.4f}%")
        print(f"   MAE Tenure:       {mae_tenure:.2f} months")
        print(f"   MAE Amount:       ₹{mae_amount:,.2f}")
    
    # Train clustering model
    print("\n👥 Training Customer Clustering...")
    X_clustering = X[y_eligibility == 1]
    
    # Find optimal clusters
    from sklearn.metrics import silhouette_score
    best_score = -1
    best_k = 5
    
    for k in range(3, 8):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_clustering)
        score = silhouette_score(X_clustering, labels_temp)
        print(f"   k={k}: Silhouette Score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    
    # Train final clustering model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_clustering)
    
    print(f"✅ Optimal clusters: {best_k} (Silhouette: {best_score:.4f})")
    
    # Save all models
    print("\n💾 Saving models...")
    joblib.dump(rf_eligibility, 'models/saved_models/eligibility_model.pkl')
    joblib.dump(rf_terms, 'models/saved_models/terms_model.pkl')
    joblib.dump(kmeans, 'models/saved_models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(encoders, 'models/saved_models/label_encoders.pkl')
    
    # Create model metrics
    model_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae_rate': mae_rate if 'mae_rate' in locals() else 1.0,
        'mae_tenure': mae_tenure if 'mae_tenure' in locals() else 5.0,
        'mae_amount': mae_amount if 'mae_amount' in locals() else 50000.0,
        'silhouette_score': best_score,
        'n_clusters': best_k,
        'feature_importance': rf_eligibility.feature_importances_.tolist()
    }
    
    joblib.dump(model_metrics, 'models/saved_models/model_metrics.pkl')
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    create_visualizations(df, X, y_eligibility, cluster_labels, model_metrics, plots_dir)
    
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📁 Models saved in: models/saved_models/")
    print(f"🧠 Eligibility Model: Random Forest (Accuracy: {accuracy:.4f})")
    print(f"💰 Terms Model: Random Forest Regressor")
    print(f"👥 Clustering: KMeans ({best_k} clusters)")
    print(f"📊 Plots: {plots_dir}/comprehensive_analysis.png")
    print("🚀 Ready to run: python app.py")
    
    return True

if __name__ == "__main__":
    train_stable_models()