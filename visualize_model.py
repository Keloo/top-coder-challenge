import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence
import joblib

def load_model(model_path: str):
    """Load the trained XGBoost model"""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
        exit(1)

def print_tree_structure(model, num_trees=3):
    """Print the structure of the first few trees in text format"""
    feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    
    for i in range(min(num_trees, model.n_estimators)):
        print(f"\n{'='*50}")
        print(f"Tree {i+1} Structure:")
        print(f"{'='*50}")
        
        # Get the tree structure
        tree = model.get_booster().get_dump()[i]
        
        # Process each line of the tree
        for line in tree.split('\n'):
            if not line:  # Skip empty lines
                continue
                
            # Count the depth by the number of tabs
            depth = line.count('\t')
            indent = '  ' * depth
            
            # Clean up the line
            line = line.strip()
            
            # Replace feature indices with names
            for idx, name in enumerate(feature_names):
                line = line.replace(f'f{idx}', name)
            
            # Add some formatting
            if 'leaf' in line:
                # Extract the leaf value
                value = float(line.split('leaf=')[1])
                print(f"{indent}Leaf: {value:.2f}")
            else:
                # Split the condition
                condition = line.split('[')[1].split(']')[0]
                print(f"{indent}If {condition}")
        
        print(f"{'='*50}\n")

def plot_feature_importance(model):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    importance = model.feature_importances_
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.barh(pos, importance[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_partial_dependence(model, X_train):
    """Plot partial dependence plots for each feature"""
    feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feature in enumerate(feature_names):
        # Calculate partial dependence
        feature_idx = i
        pdp = partial_dependence(
            model, X_train, [feature_idx],
            kind='average', grid_resolution=50
        )
        
        # Plot - pdp[0] contains the predictions, pdp[1] contains the feature values
        axes[i].plot(pdp[1][0], pdp[0].ravel())
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Partial dependence')
        axes[i].set_title(f'Partial Dependence Plot for {feature}')
    
    plt.tight_layout()
    plt.savefig('partial_dependence.png')
    plt.close()

def plot_tree_structure(model, num_trees=3):
    """Plot the structure of the first few trees"""
    for i in range(min(num_trees, model.n_estimators)):
        plt.figure(figsize=(20, 10))
        xgb.plot_tree(model, num_trees=i)
        plt.title(f'Tree {i+1} Structure')
        plt.savefig(f'tree_{i+1}_structure.png')
        plt.close()

def plot_prediction_surface(model):
    """Create 3D surface plots of predictions"""
    feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    
    # Create meshgrid for each pair of features
    for i in range(3):
        for j in range(i+1, 3):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create meshgrid
            x = np.linspace(0, 5, 50)  # days
            y = np.linspace(0, 200, 50)  # miles
            X, Y = np.meshgrid(x, y)
            
            # Calculate predictions
            Z = np.zeros_like(X)
            for k in range(len(x)):
                for l in range(len(y)):
                    # Create input array with mean values
                    input_data = np.array([[1.0, 100.0, 50.0]])  # default values
                    input_data[0, i] = X[l, k]
                    input_data[0, j] = Y[l, k]
                    Z[l, k] = model.predict(input_data)[0]
            
            # Plot surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis')
            
            # Add labels
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
            ax.set_zlabel('Predicted Reimbursement')
            ax.set_title(f'Prediction Surface: {feature_names[i]} vs {feature_names[j]}')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            plt.savefig(f'prediction_surface_{i}_{j}.png')
            plt.close()

def main():
    # Load the model
    print("Loading model...")
    model = load_model('reimbursement_model.joblib')
    
    # Load some training data for partial dependence plots
    print("Loading training data...")
    with open('public_cases.json', 'r') as f:
        import json
        data = json.load(f)
    
    X_train = np.array([
        [
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount']
        ]
        for case in data
    ])
    
    # Generate visualizations
    print("Generating visualizations...")
    
    print("Printing tree structures...")
    print_tree_structure(model)
    
    print("Plotting feature importance...")
    plot_feature_importance(model)
    
    print("Plotting partial dependence...")
    plot_partial_dependence(model, X_train)
    
    print("Plotting tree structures...")
    plot_tree_structure(model)
    
    print("Plotting prediction surfaces...")
    plot_prediction_surface(model)
    
    print("\nVisualizations complete! Check the following files:")
    print("- feature_importance.png")
    print("- partial_dependence.png")
    print("- tree_1_structure.png, tree_2_structure.png, tree_3_structure.png")
    print("- prediction_surface_0_1.png, prediction_surface_0_2.png, prediction_surface_1_2.png")

if __name__ == "__main__":
    main() 