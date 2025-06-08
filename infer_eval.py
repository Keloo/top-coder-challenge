import sys
import numpy as np
import joblib

def load_model(model_path: str):
    """
    Load the trained model
    """
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
        sys.exit(1)

def predict_reimbursement(model, days: float, miles: float, receipts: float) -> float:
    """
    Make prediction using the trained model
    """
    # Prepare input data
    X = np.array([[days, miles, receipts]])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Round to 2 decimal places
    return round(prediction, 2)

def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python infer.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        # Parse input parameters
        days = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        # Validate input
        if days < 0 or miles < 0 or receipts < 0:
            print("Error: All values must be non-negative")
            sys.exit(1)
        
        # Load model and make prediction
        model = load_model('reimbursement_model_v2.joblib')
        result = predict_reimbursement(model, days, miles, receipts)
        
        # Print result
        print(f"{result:.2f}")
        
    except ValueError:
        print("Error: All arguments must be valid numbers")
        sys.exit(1)

if __name__ == "__main__":
    main() 