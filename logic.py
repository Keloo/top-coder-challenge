#!/usr/bin/env python3

import sys
import numpy as np
import json
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations_with_replacement

def generate_polynomial_terms(degree: int) -> List[Tuple[int, int, int]]:
    """
    Generate all possible terms for a polynomial of given degree with 3 variables.
    Each term is represented as (power_of_days, power_of_miles, power_of_receipts).
    
    Args:
        degree: Maximum degree of the polynomial
    
    Returns:
        List of tuples representing the powers of each variable in each term
    """
    terms = []
    for d in range(degree + 1):
        for m in range(degree + 1):
            for r in range(degree + 1):
                if d + m + r <= degree:
                    terms.append((d, m, r))
    return terms

def calculate_term_value(days: float, miles: float, receipts: float, term: Tuple[int, int, int]) -> float:
    """
    Calculate the value of a single term in the polynomial.
    """
    d_pow, m_pow, r_pow = term
    return (days ** d_pow) * (miles ** m_pow) * (receipts ** r_pow)

def calculate_reimbursement(days, miles, receipts, coefficients, terms):
    """
    Calculate reimbursement using coefficients for a polynomial of degree N.
    
    Args:
        days: Number of days
        miles: Miles traveled
        receipts: Total receipts amount
        coefficients: List of coefficients for each term
        terms: List of terms (power combinations)
    """
    total = sum(
        coef * calculate_term_value(days, miles, receipts, term)
        for coef, term in zip(coefficients, terms)
    )
    return round(total, 2)

def print_and_plot_function(coefficients, terms, degree):
    """
    Print the learned function and create a 3D plot showing how reimbursement varies
    with days and miles (holding receipts constant at their mean value).
    """
    # Print the function
    print("\nLearned Function (degree", degree, "polynomial):")
    function_str = "f(days, miles, receipts) = "
    terms_str = []
    
    for coef, (d_pow, m_pow, r_pow) in zip(coefficients, terms):
        if abs(coef) < 1e-10:  # Skip very small coefficients
            continue
            
        term_str = f"{coef:.4f}"
        if d_pow > 0:
            term_str += f"*days^{d_pow}"
        if m_pow > 0:
            term_str += f"*miles^{m_pow}"
        if r_pow > 0:
            term_str += f"*receipts^{r_pow}"
        terms_str.append(term_str)
    
    print(function_str + " + ".join(terms_str))
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of days and miles values
    days = np.linspace(0, 5, 50)
    miles = np.linspace(0, 200, 50)
    days_grid, miles_grid = np.meshgrid(days, miles)
    
    # Use mean receipts value from training data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    mean_receipts = np.mean([case['input']['total_receipts_amount'] for case in data])
    
    # Calculate reimbursement for each point in the grid
    reimbursement = np.zeros_like(days_grid)
    for i in range(len(days)):
        for j in range(len(miles)):
            reimbursement[j, i] = calculate_reimbursement(
                days[i], miles[j], mean_receipts, coefficients, terms
            )
    
    # Plot the surface
    surf = ax.plot_surface(days_grid, miles_grid, reimbursement, cmap='viridis')
    
    # Add labels and title
    ax.set_xlabel('Days')
    ax.set_ylabel('Miles')
    ax.set_zlabel('Reimbursement ($)')
    ax.set_title(f'Reimbursement Function (degree {degree}, receipts fixed at ${mean_receipts:.2f})')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Save the plot
    plt.savefig(f'reimbursement_function_degree_{degree}.png')
    plt.close()

def solve_coefficients(cases: List[Tuple[float, float, float, float]], degree: int) -> Tuple[List[float], List[Tuple[int, int, int]]]:
    """
    Solve for coefficients in a polynomial of given degree.
    
    Args:
        cases: List of training cases
        degree: Maximum degree of the polynomial
    
    Returns:
        Tuple of (coefficients, terms) where terms are the power combinations
    """
    # Generate all possible terms
    terms = generate_polynomial_terms(degree)
    
    # Create feature matrix with all terms and ensure float64 type
    X = np.array([
        [float(calculate_term_value(d, m, r, term)) for term in terms]
        for d, m, r, _ in cases
    ], dtype=np.float64)
    
    # Ensure y is also float64
    y = np.array([float(o) for _, _, _, o in cases], dtype=np.float64)
    
    # Solve the linear system using numpy's least squares
    coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    return coefficients, terms

def load_cases_from_json(file_path: str, num_cases: int = None) -> List[Tuple[float, float, float, float]]:
    """
    Load cases from a JSON file and optionally select a random subset.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert JSON data to list of tuples
    all_cases = [
        (
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount'],
            case['expected_output']
        )
        for case in data
    ]
    
    # Return all cases or a random subset
    if num_cases is None:
        return all_cases
    return random.sample(all_cases, min(num_cases, len(all_cases)))

def find_worst_cases(coefficients, terms, num_cases: int):
    """
    Find and print the N cases with the largest prediction errors.
    """
    # Load all cases
    cases = load_cases_from_json('public_cases.json')
    
    # Calculate errors for all cases
    errors = []
    for days, miles, receipts, expected in cases:
        predicted = calculate_reimbursement(days, miles, receipts, coefficients, terms)
        error = abs(predicted - expected)
        errors.append((days, miles, receipts, expected, predicted, error))
    
    # Sort by error and get worst N cases
    errors.sort(key=lambda x: x[5], reverse=True)
    worst_cases = errors[:num_cases]
    
    # Print results
    print(f"\nTop {num_cases} worst cases:")
    print("Days | Miles | Receipts | Expected | Predicted | Error")
    print("-" * 65)
    for days, miles, receipts, expected, predicted, error in worst_cases:
        print(f"{days:4.1f} | {miles:5.1f} | ${receipts:8.2f} | ${expected:8.2f} | ${predicted:8.2f} | ${error:6.2f}")

def main():
    # Parse command line arguments
    args = sys.argv[1:]
    print_worst = None
    
    # Check for --print_worst option
    if len(args) >= 2 and args[-2] == "--print_worst":
        try:
            print_worst = int(args[-1])
            args = args[:-2]  # Remove the --print_worst option and its value
        except ValueError:
            print("Error: --print_worst must be followed by a number")
            sys.exit(1)
    
    # If only --print_worst was provided
    if len(args) == 0 and print_worst is not None:
        try:
            # Load training data and find coefficients
            cases = load_cases_from_json('public_cases.json', 1000)
            coefficients, terms = solve_coefficients(cases, 4)  # Using degree 4
            
            # Find and print worst cases
            find_worst_cases(coefficients, terms, print_worst)
            return
        except FileNotFoundError:
            print("Error: public_cases.json not found")
            sys.exit(1)
    
    # Otherwise, require the three input parameters
    if len(args) != 3:
        print("Usage: python3 logic.py <trip_duration_days> <miles_traveled> <total_receipts_amount> [--print_worst N]")
        print("   or: python3 logic.py --print_worst N")
        sys.exit(1)
    
    try:
        days = float(args[0])
        miles = float(args[1])
        receipts = float(args[2])
        degree = 1
        
        if days < 0 or miles < 0 or receipts < 0:
            print("Error: All values must be non-negative")
            sys.exit(1)
        
        # Load training data and find coefficients
        cases = load_cases_from_json('public_cases.json', 1000)
        coefficients, terms = solve_coefficients(cases, degree)
        
        # Calculate using learned coefficients
        result = calculate_reimbursement(days, miles, receipts, coefficients, terms)
        print(f"{result:.2f}")
        
        # If --print_worst was specified, find and print worst cases
        if print_worst is not None:
            find_worst_cases(coefficients, terms, print_worst)
        
    except ValueError:
        print("Error: All arguments must be valid numbers")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: public_cases.json not found")
        sys.exit(1)

if __name__ == "__main__":
    main()