# Dynamic Loan Origination System Guide

## Overview

This loan origination system has been upgraded to be fully dynamic with multiple deep learning models (CNN, ANN, RNN) and ensemble methods. Everything is now calculated dynamically based on the data.

## Key Features

### 1. **Dynamic Data Processing**
- Automatically detects column types (numeric, categorical, datetime, text)
- Creates interaction features dynamically
- Handles missing values intelligently
- Scales and encodes features automatically

### 2. **Multiple Model Architecture**
The system now uses an ensemble of models:

- **Random Forest** (74.0% accuracy) - Traditional ML baseline
- **Gradient Boosting** (75.4% accuracy) - Best performing model
- **Artificial Neural Network (ANN)** (69.3% accuracy) - Deep learning model
- **Convolutional Neural Network (CNN)** (69.3% accuracy) - 1D CNN for tabular data
- **K-Means Clustering** - For customer segmentation (3 optimal clusters)

### 3. **Fully Dynamic Processing**
- No hardcoded features or assumptions
- Adapts to any CSV dataset structure
- Automatically identifies target variable
- Creates features based on data patterns

## Installation & Setup

### For macOS (Apple Silicon)

```bash
# Install dependencies
pip install -r requirements-mac.txt

# If TensorFlow issues occur, use:
pip install tensorflow-macos==2.16.2
pip install tensorflow-metal==1.2.0
```

### Training Models

```bash
# Use the stable dynamic training script
python train_dynamic_stable.py

# This will:
# 1. Load your dataset from data/Insurance_Enhanced.csv
# 2. Automatically prepare features
# 3. Train all models (RF, GB, ANN, CNN, KMeans)
# 4. Save models to models/saved_models/
# 5. Create visualizations in static/plots/
```

## Running the Application

```bash
# Start the Flask app
python app.py

# The app will:
# - Load all trained models dynamically
# - Use ensemble predictions for better accuracy
# - Provide personalized loan offers
```

## How the Dynamic System Works

### 1. Data Loading
```python
# Automatically detects and loads any CSV in data/
data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
df = pd.read_csv(data_files[0])
```

### 2. Feature Engineering
The system creates features dynamically:
- **Numeric features**: Statistical features, z-scores, binning
- **Categorical features**: One-hot or label encoding based on cardinality
- **Interaction features**: Top feature combinations
- **Datetime features**: Year, month, day, weekday, etc.

### 3. Model Training
- Models are trained in parallel where possible
- Automatic hyperparameter selection
- Cross-validation for robust performance

### 4. Prediction Pipeline
```python
# The dynamic predictor uses ensemble voting
predictions = []
for model in [rf, gb, ann, cnn]:
    predictions.append(model.predict(features))
final_prediction = weighted_average(predictions)
```

## Model Performance

| Model | Accuracy | Use Case |
|-------|----------|----------|
| Gradient Boosting | 75.4% | Best overall performance |
| Random Forest | 74.0% | Stable, interpretable |
| ANN | 69.3% | Complex patterns |
| CNN | 69.3% | Feature interactions |

## API Endpoints

### `/predict_loan` (POST)
Dynamically processes input and returns:
- Eligibility probability
- Loan terms (interest rate, tenure, amount)
- Customer persona
- Three personalized loan offers

### `/get_analytics` (GET)
Returns:
- Model performance metrics
- Cluster information
- Feature importance

## Customization

### Using Different Datasets
1. Place your CSV in the `data/` folder
2. Ensure it has a target column (e.g., 'loan_status', 'approved')
3. Run `python train_dynamic_stable.py`

### Switching Between Models
In `app.py`, set:
```python
USE_DYNAMIC_MODELS = True  # Use ensemble of deep learning models
USE_DYNAMIC_MODELS = False # Use original simple models
```

### Adding New Models
1. Add model definition in `train_dynamic_stable.py`
2. Train and save the model
3. Update `DynamicLoanPredictor` to load it

## Troubleshooting

### TensorFlow Memory Issues on macOS
The stable training script disables GPU to avoid memory issues:
```python
tf.config.set_visible_devices([], 'GPU')
```

### Model Not Loading
Check that all required files exist in `models/saved_models/`:
- `rf_model.pkl`
- `gb_model.pkl`
- `ann_model.h5`
- `cnn_model.h5`
- `kmeans_model.pkl`
- `data_processor.pkl`

### Low Accuracy
- Ensure sufficient training data (>1000 samples)
- Check for class imbalance
- Try different feature engineering approaches

## Future Enhancements

1. **AutoML Integration**: Automatic model selection
2. **Real-time Learning**: Update models with new data
3. **Explainable AI**: SHAP/LIME for model interpretability
4. **More Models**: XGBoost, LightGBM, Transformer models
5. **Feature Store**: Centralized feature management

## Support

For issues or questions:
1. Check model logs in the console
2. Verify all dependencies are installed
3. Ensure data format is correct
4. Try the stable training script if TensorFlow fails 