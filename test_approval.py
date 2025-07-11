"""
Test script for loan approval with high-quality applicant
"""

import requests
import json

# High-quality applicant data
test_data = {
    "PI_AGE": 40,
    "PI_ANNUAL_INCOME": 2000000,  # 20 lakhs
    "SUM_ASSURED": 1000000,       # 10 lakhs
    "PI_GENDER": "M",
    "PI_OCCUPATION": "Government Employee",
    "ZONE": "Metro",
    "PAYMENT_MODE": "Monthly",
    "EARLY_NON": "No",
    "MEDICAL_NONMED": "Medical",
    "PI_STATE": "Delhi"
}

BASE_URL = "http://localhost:5000"

def test_high_quality_applicant():
    print("🎯 Testing High-Quality Loan Applicant")
    print("=" * 50)
    
    print("📝 Applicant Profile:")
    print(f"   Age: {test_data['PI_AGE']} years")
    print(f"   Income: ₹{test_data['PI_ANNUAL_INCOME']:,} per year")
    print(f"   Sum Assured: ₹{test_data['SUM_ASSURED']:,}")
    print(f"   Occupation: {test_data['PI_OCCUPATION']}")
    print(f"   Zone: {test_data['ZONE']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_loan",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print(f"\n✅ Prediction Result:")
                print(f"   Eligible: {result.get('eligible')}")
                print(f"   Probability: {result.get('probability', 0):.4f}")
                
                if result.get('eligible'):
                    terms = result.get('loan_terms', {})
                    print(f"\n💰 Loan Terms:")
                    print(f"   Interest Rate: {terms.get('rate_of_interest', 0)}%")
                    print(f"   Tenure: {terms.get('tenure_months', 0)} months")
                    print(f"   Sanctioned Amount: ₹{terms.get('sanctioned_amount', 0):,}")
                    
                    persona = result.get('persona', 0)
                    print(f"\n👤 Customer Persona: {persona}")
                    
                    offers = result.get('offers', [])
                    if offers:
                        print(f"\n🎁 Loan Offers ({len(offers)} available):")
                        for i, offer in enumerate(offers, 1):
                            print(f"\n   Offer {i}: {offer.get('type', 'Unknown')}")
                            print(f"   - Rate: {offer.get('rate_of_interest', 0)}%")
                            print(f"   - Tenure: {offer.get('tenure_months', 0)} months")
                            print(f"   - Amount: ₹{offer.get('sanctioned_amount', 0):,}")
                            print(f"   - EMI: ₹{offer.get('emi', 0):,}")
                            print(f"   - Total Interest: ₹{offer.get('total_interest', 0):,}")
                            
                            if 'recommendation_score' in offer:
                                print(f"   - Score: {offer['recommendation_score']}/100")
                else:
                    print(f"\n❌ Loan not approved")
                    print(f"   Reason: {result.get('message', 'Unknown')}")
            else:
                print(f"❌ Prediction failed: {result.get('message')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Connection error: {e}")

def test_analytics():
    """Test the analytics endpoint"""
    print(f"\n\n📊 Model Analytics")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/get_analytics")
        if response.status_code == 200:
            data = response.json()
            print("✅ Analytics retrieved successfully")
            
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"\n🤖 Model Information:")
                print(f"   Models loaded: {metrics.get('models_loaded', 0)}")
                print(f"   All three models active: {metrics.get('all_three_models_active', False)}")
                
                # Show individual model performance
                for key, value in metrics.items():
                    if key.endswith('_accuracy'):
                        model_name = key.replace('_accuracy', '')
                        print(f"   {model_name}: {value:.4f} accuracy")
        else:
            print(f"❌ Analytics error: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Analytics error: {e}")

if __name__ == "__main__":
    test_high_quality_applicant()
    test_analytics()
    
    print("\n\n🔍 Check the Flask app console for detailed ensemble predictions!")
    print("You should see output like:")
    print("🔮 Ensemble prediction using ['CNN', 'ANN', 'RNN', 'RF']: 0.XXXX") 