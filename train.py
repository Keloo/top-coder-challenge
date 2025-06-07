import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

def load_and_prepare_data(file_path: str, test_size: float = 0.3):
    """
    Load data from JSON and split into train/test sets
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract features and target
    X = np.array([
        [
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount']
        ]
        for case in data
    ])
    
    y = np.array([case['expected_output'] for case in data])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train XGBoost model with optimized parameters
    """
    # Define model parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.01,  # Reduced from 0.1 to prevent overfitting
        'max_depth': 100,         # Slightly increased to allow more complex patterns
        'min_child_weight': 3,  # Increased to prevent overfitting on small groups
        'subsample': 0.8,       # Kept the same for good balance
        'colsample_bytree': 1, # We have only 3 features, so we can afford to use all of them
        'n_estimators': 1000,   # Increased from 100 to 1000
        'random_state': 69,
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set
    """
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    
    print("\nModel Performance on Test Set:")
    print(f"Root Mean Square Error: ${rmse:.2f}")
    print(f"Mean Absolute Error: ${mae:.2f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    importance = model.feature_importances_
    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.4f}")

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data('public_cases.json')
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    print("\nSaving model...")
    joblib.dump(model, 'reimbursement_model.joblib')
    print("Model saved as 'reimbursement_model.joblib'")

if __name__ == "__main__":
    main() 