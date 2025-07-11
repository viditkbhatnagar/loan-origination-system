"""
Test script to verify the dynamic prediction system with CNN, ANN, and RNN
"""

import requests
import json
import time

# API endpoint
BASE_URL = "http://localhost:5000"

def test_model_status():
    """Check if all models are loaded"""
    print("🔍 Testing Model Status...")
    print("-" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/get_analytics")
        if response.status_code == 200:
            data = response.json()
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"✅ Models loaded: {metrics.get('models_loaded', 0)}")
                print(f"✅ Model types: {metrics.get('model_types', [])}")
                print(f"✅ All three models active: {metrics.get('all_three_models_active', False)}")
                
                # Show model weights
                if 'model_weights' in metrics:
                    print("\n📊 Model Weights:")
                    for model, weight in metrics['model_weights'].items():
                        print(f"   {model}: {weight:.3f}")
            else:
                print("⚠️  No metrics available")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("   Make sure the Flask app is running!")

def test_loan_prediction():
    """Test loan prediction with sample data"""
    print("\n\n🎯 Testing Loan Prediction...")
    print("-" * 50)
    
    # Sample loan application data
    test_data = {
        "PI_AGE": 35,
        "PI_ANNUAL_INCOME": 750000,
        "SUM_ASSURED": 500000,
        "PI_GENDER": "M",
        "PI_OCCUPATION": "Salaried",
        "ZONE": "Metro",
        "PAYMENT_MODE": "Monthly",
        "EARLY_NON": "No",
        "MEDICAL_NONMED": "Medical",
        "PI_STATE": "Delhi"
    }
    
    print("📝 Test Data:")
    print(f"   Age: {test_data['PI_AGE']}")
    print(f"   Income: ₹{test_data['PI_ANNUAL_INCOME']:,}")
    print(f"   Sum Assured: ₹{test_data['SUM_ASSURED']:,}")
    
    try:
        # Make prediction request
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict_loan",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print(f"\n✅ Prediction successful! (Time: {end_time - start_time:.2f}s)")
                print(f"   Eligible: {result.get('eligible')}")
                print(f"   Probability: {result.get('probability', 0):.4f}")
                
                if 'loan_terms' in result:
                    terms = result['loan_terms']
                    print(f"\n💰 Loan Terms:")
                    print(f"   Interest Rate: {terms['rate_of_interest']}%")
                    print(f"   Tenure: {terms['tenure_months']} months")
                    print(f"   Amount: ₹{terms['sanctioned_amount']:,}")
                
                if 'offers' in result and len(result['offers']) > 0:
                    print(f"\n🎁 Generated {len(result['offers'])} loan offers")
                    for i, offer in enumerate(result['offers']):
                        print(f"\n   Offer {i+1}: {offer['type']}")
                        print(f"   EMI: ₹{offer['emi']:,}")
                        print(f"   Interest: {offer['rate_of_interest']}%")
            else:
                print(f"❌ Prediction failed: {result.get('message')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Connection error: {e}")

def test_multiple_predictions():
    """Test multiple predictions to see ensemble working"""
    print("\n\n🔄 Testing Multiple Predictions...")
    print("-" * 50)
    
    test_cases = [
        {"name": "Young Professional", "age": 25, "income": 400000},
        {"name": "Mid-Career", "age": 40, "income": 1000000},
        {"name": "Senior Executive", "age": 55, "income": 2000000}
    ]
    
    for test in test_cases:
        data = {
            "PI_AGE": test["age"],
            "PI_ANNUAL_INCOME": test["income"],
            "SUM_ASSURED": test["income"] * 0.5,
            "PI_GENDER": "M",
            "PI_OCCUPATION": "Salaried",
            "ZONE": "Metro",
            "PAYMENT_MODE": "Monthly",
            "EARLY_NON": "No",
            "MEDICAL_NONMED": "Medical",
            "PI_STATE": "Delhi"
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict_loan",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                prob = result.get('probability', 0)
                print(f"\n{test['name']}: {prob:.4f} probability")
                
                # Check console output for ensemble details
                print("   (Check app console for ensemble details)")
        
        except Exception as e:
            print(f"❌ Error for {test['name']}: {e}")
        
        time.sleep(0.5)  # Small delay between requests

if __name__ == "__main__":
    print("🚀 Dynamic Loan Prediction System Test")
    print("=" * 60)
    print("Make sure the Flask app is running on http://localhost:5000")
    print("=" * 60)
    
    # Run tests
    test_model_status()
    test_loan_prediction()
    test_multiple_predictions()
    
    print("\n\n✅ Testing complete!")
    print("Check the Flask app console for detailed ensemble predictions.") 