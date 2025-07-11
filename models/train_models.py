import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_preprocessing import DataPreprocessor
from config import Config

def create_visualizations(df, X_scaled, y_eligibility, cluster_labels, model_metrics):
    """Create comprehensive visualizations"""
    
    # Create static directory for plots
    plots_dir = 'static/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Data Distribution Analysis
    plt.figure(figsize=(20, 15))
    
    # Age distribution by eligibility
    plt.subplot(3, 4, 1)
    df.boxplot(column='PI_AGE', by='POLICY_STATUS', ax=plt.gca())
    plt.title('Age Distribution by Loan Status')
    plt.suptitle('')
    
    # Income distribution
    plt.subplot(3, 4, 2)
    df.boxplot(column='PI_ANNUAL_INCOME', by='POLICY_STATUS', ax=plt.gca())
    plt.title('Income Distribution by Loan Status')
    plt.suptitle('')
    
    # Sum assured distribution
    plt.subplot(3, 4, 3)
    df.boxplot(column='SUM_ASSURED', by='POLICY_STATUS', ax=plt.gca())
    plt.title('Sum Assured by Loan Status')
    plt.suptitle('')
    
    # Gender distribution
    plt.subplot(3, 4, 4)
    gender_status = pd.crosstab(df['PI_GENDER'], df['POLICY_STATUS'])
    gender_status.plot(kind='bar', ax=plt.gca())
    plt.title('Gender vs Loan Status')
    plt.xticks(rotation=45)
    
    # Occupation distribution
    plt.subplot(3, 4, 5)
    occupation_counts = df['PI_OCCUPATION'].value_counts().head(10)
    occupation_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Top 10 Occupations')
    plt.xticks(rotation=45)
    
    # Zone distribution
    plt.subplot(3, 4, 6)
    zone_status = pd.crosstab(df['ZONE'], df['POLICY_STATUS'])
    zone_status.plot(kind='bar', ax=plt.gca())
    plt.title('Zone vs Loan Status')
    plt.xticks(rotation=45)
    
    # Rate of interest distribution
    plt.subplot(3, 4, 7)
    plt.hist(df['rate_of_interest'], bins=20, alpha=0.7, color='skyblue')
    plt.title('Rate of Interest Distribution')
    plt.xlabel('Interest Rate (%)')
    
    # Tenure distribution
    plt.subplot(3, 4, 8)
    plt.hist(df['tenure_months'], bins=15, alpha=0.7, color='lightgreen')
    plt.title('Tenure Distribution')
    plt.xlabel('Tenure (months)')
    
    # Sanctioned amount distribution
    plt.subplot(3, 4, 9)
    plt.hist(df['sanctioned_amount'], bins=20, alpha=0.7, color='orange')
    plt.title('Sanctioned Amount Distribution')
    plt.xlabel('Amount (₹)')
    
    # Correlation heatmap
    plt.subplot(3, 4, 10)
    numeric_cols = ['PI_AGE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED', 'rate_of_interest', 'tenure_months', 'sanctioned_amount']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=plt.gca())
    plt.title('Feature Correlation Matrix')
    
    # Model performance metrics
    plt.subplot(3, 4, 11)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [model_metrics['accuracy'], model_metrics['precision'], model_metrics['recall'], model_metrics['f1']]
    plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Feature importance (if available)
    plt.subplot(3, 4, 12)
    if 'feature_importance' in model_metrics:
        feature_names = ['Age', 'Income', 'Sum Assured', 'Gender', 'Occupation', 'Zone', 'Payment', 'Early', 'Medical', 'State']
        importance = model_metrics['feature_importance'][:len(feature_names)]
        plt.barh(feature_names, importance)
        plt.title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/data_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Cluster Analysis
    plt.figure(figsize=(16, 12))
    
    # Cluster visualization (2D projection using PCA)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Customer Clusters (PCA Projection)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Cluster characteristics
    cluster_df = pd.DataFrame(X_scaled)
    cluster_df['cluster'] = cluster_labels
    
    for i in range(2, 6):
        plt.subplot(2, 3, i)
        feature_idx = i - 2
        if feature_idx < X_scaled.shape[1]:
            cluster_df.boxplot(column=feature_idx, by='cluster', ax=plt.gca())
            feature_names = ['Age', 'Income', 'Sum Assured', 'Gender', 'Occupation']
            if feature_idx < len(feature_names):
                plt.title(f'{feature_names[feature_idx]} by Cluster')
            plt.suptitle('')
    
    # Cluster distribution
    plt.subplot(2, 3, 6)
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    plt.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Customer Distribution Across Clusters')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/cluster_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualizations saved to static/plots/")

def train_advanced_models():
    """Train advanced deep learning models with comprehensive analysis"""
    
    print("🚀 Starting Advanced Model Training with TensorFlow")
    print("=" * 60)
    
    # Load dataset
    if not os.path.exists('data/Insurance_Enhanced.csv'):
        print("❌ Error: Insurance_Enhanced.csv not found in data/ directory")
        return False
    
    df = pd.read_csv('data/Insurance_Enhanced.csv')
    print(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare data with advanced preprocessing
    X, y_eligibility, y_terms = preprocessor.prepare_training_data(df)
    print(f"🔧 Features prepared: {X.shape}")
    print(f"📈 Class distribution: {np.bincount(y_eligibility)}")
    
    # Advanced train-test split with stratification
    X_train, X_test, y_elig_train, y_elig_test, y_terms_train, y_terms_test = train_test_split(
        X, y_eligibility, y_terms, test_size=0.2, random_state=42, stratify=y_eligibility
    )
    
    # Create models directory
    os.makedirs(Config.MODELS_FOLDER, exist_ok=True)
    
    print("\n🧠 Training Advanced ANN for Eligibility Prediction...")
    print("-" * 50)
    
    # Advanced ANN Architecture
    ann_model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Advanced optimizer and compilation
    ann_model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("🏗️  ANN Architecture:")
    ann_model.summary()
    
    # Advanced callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, monitor='val_loss'),
        ModelCheckpoint(os.path.join(Config.MODELS_FOLDER, 'best_ann_model.h5'), 
                       save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train ANN with validation
    history_ann = ann_model.fit(
        X_train, y_elig_train,
        validation_data=(X_test, y_elig_test),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=callbacks
    )
    
    # Comprehensive evaluation
    y_pred_proba = ann_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    ann_accuracy = accuracy_score(y_elig_test, y_pred)
    ann_precision = precision_score(y_elig_test, y_pred)
    ann_recall = recall_score(y_elig_test, y_pred)
    ann_f1 = f1_score(y_elig_test, y_pred)
    
    print(f"\n📊 ANN Model Performance:")
    print(f"   Accuracy:  {ann_accuracy:.4f}")
    print(f"   Precision: {ann_precision:.4f}")
    print(f"   Recall:    {ann_recall:.4f}")
    print(f"   F1-Score:  {ann_f1:.4f}")
    
    # Save final ANN model
    ann_model.save(os.path.join(Config.MODELS_FOLDER, 'ann_model.h5'))
    
    # Train Advanced RNN for loan terms
    print("\n🔮 Training Advanced RNN for Loan Terms Prediction...")
    print("-" * 50)
    
    # Filter to eligible customers only
    eligible_mask = y_elig_train == 1
    X_train_eligible = X_train[eligible_mask]
    y_terms_train_eligible = y_terms_train[eligible_mask]
    
    eligible_mask_test = y_elig_test == 1
    X_test_eligible = X_test[eligible_mask_test]
    y_terms_test_eligible = y_terms_test[eligible_mask_test]
    
    print(f"🎯 Training RNN on {len(X_train_eligible)} eligible customers")
    
    # Normalize target variables
    from sklearn.preprocessing import StandardScaler
    terms_scaler = StandardScaler()
    y_terms_train_scaled = terms_scaler.fit_transform(y_terms_train_eligible)
    y_terms_test_scaled = terms_scaler.transform(y_terms_test_eligible)
    
    # Reshape for RNN (sequence length = 1)
    X_train_rnn = X_train_eligible.reshape(X_train_eligible.shape[0], 1, X_train_eligible.shape[1])
    X_test_rnn = X_test_eligible.reshape(X_test_eligible.shape[0], 1, X_test_eligible.shape[1])
    
    # Advanced RNN Architecture
    rnn_model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(1, X_train_eligible.shape[1])),
        Dropout(0.3),
        
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3)  # rate, tenure, amount
    ])
    
    rnn_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    print("🏗️  RNN Architecture:")
    rnn_model.summary()
    
    # Train RNN
    history_rnn = rnn_model.fit(
        X_train_rnn, y_terms_train_scaled,
        validation_data=(X_test_rnn, y_terms_test_scaled),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10)
        ]
    )
    
    # Evaluate RNN
    y_pred_terms_scaled = rnn_model.predict(X_test_rnn)
    y_pred_terms = terms_scaler.inverse_transform(y_pred_terms_scaled)
    
    mae_rate = np.mean(np.abs(y_pred_terms[:, 0] - y_terms_test_eligible[:, 0]))
    mae_tenure = np.mean(np.abs(y_pred_terms[:, 1] - y_terms_test_eligible[:, 1]))
    mae_amount = np.mean(np.abs(y_pred_terms[:, 2] - y_terms_test_eligible[:, 2]))
    
    print(f"\n📊 RNN Model Performance:")
    print(f"   MAE - Interest Rate: {mae_rate:.4f}%")
    print(f"   MAE - Tenure:       {mae_tenure:.2f} months")
    print(f"   MAE - Amount:       ₹{mae_amount:,.2f}")
    
    # Save RNN model and scaler
    rnn_model.save(os.path.join(Config.MODELS_FOLDER, 'rnn_model.h5'))
    joblib.dump(terms_scaler, os.path.join(Config.MODELS_FOLDER, 'terms_scaler.pkl'))
    
    # Advanced KMeans Clustering with optimal cluster selection
    print("\n👥 Training Advanced KMeans Clustering...")
    print("-" * 50)
    
    X_clustering = X[y_eligibility == 1]
    print(f"🎯 Clustering {len(X_clustering)} eligible customers")
    
    # Find optimal number of clusters
    silhouette_scores = []
    inertias = []
    K_range = range(3, 10)
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels_temp = kmeans_temp.fit_predict(X_clustering)
        silhouette_avg = silhouette_score(X_clustering, cluster_labels_temp)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans_temp.inertia_)
        print(f"   k={k}: Silhouette Score = {silhouette_avg:.4f}")
    
    # Select best k
    best_k = K_range[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)
    
    print(f"\n🎯 Optimal clusters: {best_k} (Silhouette Score: {best_silhouette:.4f})")
    
    # Train final KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_clustering)
    
    # Cluster analysis
    print(f"\n📊 Cluster Distribution:")
    for i in range(best_k):
        cluster_size = np.sum(cluster_labels == i)
        print(f"   Cluster {i}: {cluster_size} customers ({cluster_size/len(cluster_labels)*100:.1f}%)")
    
    # Save models and metadata
    joblib.dump(kmeans, os.path.join(Config.MODELS_FOLDER, 'kmeans_model.pkl'))
    joblib.dump(preprocessor.scaler, os.path.join(Config.MODELS_FOLDER, 'scaler.pkl'))
    joblib.dump(preprocessor.label_encoders, os.path.join(Config.MODELS_FOLDER, 'label_encoders.pkl'))
    
    # Create comprehensive model metrics
    model_metrics = {
        'accuracy': ann_accuracy,
        'precision': ann_precision,
        'recall': ann_recall,
        'f1': ann_f1,
        'mae_rate': mae_rate,
        'mae_tenure': mae_tenure,
        'mae_amount': mae_amount,
        'silhouette_score': best_silhouette,
        'n_clusters': best_k
    }
    
    # Save model metrics
    joblib.dump(model_metrics, os.path.join(Config.MODELS_FOLDER, 'model_metrics.pkl'))
    
    # Create visualizations
    print("\n📈 Creating Comprehensive Visualizations...")
    create_visualizations(df, X, y_eligibility, cluster_labels, model_metrics)
    
    print("\n" + "=" * 60)
    print("🎉 ADVANCED MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Models saved in: {Config.MODELS_FOLDER}")
    print(f"🧠 ANN Model: ann_model.h5 (Accuracy: {ann_accuracy:.4f})")
    print(f"🔮 RNN Model: rnn_model.h5 (MAE Rate: {mae_rate:.4f})")
    print(f"👥 KMeans Model: kmeans_model.pkl ({best_k} clusters)")
    print(f"📊 Visualizations: static/plots/")
    print("🚀 Ready to run: python app.py")
    
    return True

if __name__ == "__main__":
    train_advanced_models()