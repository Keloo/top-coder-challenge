def calculate_reimbursement_v2(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculates reimbursement based on a complex, multi-formula system.
    This model reflects competing calculation methods and selection rules,
    which is common in financial rule engines.
    """
    
    # Define primary coefficients that change based on trip duration
    if trip_duration_days <= 2:
        per_diem_rate = 85.0
        mileage_rate = 0.65
        expense_receipt_factor = 0.60
        expense_mileage_bonus = 0.25
        expense_base = 150.0
        high_receipt_threshold = 400.0
    elif trip_duration_days <= 7:
        per_diem_rate = 75.0
        mileage_rate = 0.68
        expense_receipt_factor = 0.75
        expense_mileage_bonus = 0.35
        expense_base = 250.0
        high_receipt_threshold = 800.0
    else: # trip_duration_days >= 8
        per_diem_rate = 65.0
        mileage_rate = 0.72
        expense_receipt_factor = 0.80
        expense_mileage_bonus = 0.45
        expense_base = 350.0
        high_receipt_threshold = 1000.0

    # --- Step 1: Calculate potential reimbursements using different methods ---

    # Method 1: Standard Per-Diem/Mileage Allowance
    # This is the baseline for "normal" trips.
    standard_allowance = (trip_duration_days * per_diem_rate) + \
                         (miles_traveled * mileage_rate) + \
                         total_receipts_amount

    # Method 2: Expense-Account Reimbursement
    # This is for trips where receipts are the primary cost driver.
    expense_account_calc = expense_base + \
                           (total_receipts_amount * expense_receipt_factor) + \
                           (miles_traveled * expense_mileage_bonus)

    # --- Step 2: Apply the complex business rules to select and adjust the final amount ---

    final_reimbursement = 0.0

    # Rule 1: Extreme Mileage Override (Road Warrior)
    # This rule takes precedence over almost everything else.
    if miles_traveled > 500 and (miles_traveled / trip_duration_days) > 200:
        # A very simple formula for high-intensity travel
        final_reimbursement = 400 + (miles_traveled * 0.55) + (total_receipts_amount * 0.25)
    
    # Rule 2: Extreme Receipt Cap (Sanity Check)
    # If receipts are incredibly high, they are factored down significantly.
    elif total_receipts_amount > 2000 and total_receipts_amount > trip_duration_days * 500:
        final_reimbursement = total_receipts_amount * 0.55 + (miles_traveled * 0.30) + (trip_duration_days * 50)
    
    # Rule 3: High-Receipt "Expense Account" Logic
    # If receipts are above the threshold for the trip duration, choose the *higher*
    # of the two main calculation methods. This rewards well-documented high spending.
    elif total_receipts_amount > high_receipt_threshold:
        final_reimbursement = max(standard_allowance, expense_account_calc)
        
        # Add a bonus for long, expensive trips
        if trip_duration_days > 10 and total_receipts_amount > 1500:
            final_reimbursement += 200.0

    # Rule 4: Standard Trip Logic
    # For all other cases, use the standard allowance.
    else:
        final_reimbursement = standard_allowance

    # Final Rule: A hard cap on short trips to prevent abuse.
    # The reimbursement for a short trip cannot exceed a certain reasonable limit.
    if trip_duration_days <= 2 and final_reimbursement > 1600:
        # If the calculation is very high, it might be capped or a different rule applies.
        # Let's use the smaller of the two calculations as a more conservative estimate.
        final_reimbursement = min(standard_allowance, expense_account_calc)


    return round(final_reimbursement, 2)

def run_test_cases():
    """Run test cases from public_cases.json"""
    import json
    import os

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'public_cases.json')

    try:
        with open(json_path, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return

    print("\nRunning test cases:")
    print("=" * 50)
    for i, case in enumerate(test_cases, 1):
        inputs = case["input"]
        calculated_output = calculate_reimbursement_v2(
            inputs["trip_duration_days"],
            inputs["miles_traveled"],
            inputs["total_receipts_amount"]
        )
        print(f"\nTest Case {i}:")
        print(f"Input: {inputs}")
        print(f"Expected: ${case['expected_output']:.2f}")
        print(f"Calculated: ${calculated_output:.2f}")
        print(f"Difference: ${abs(calculated_output - case['expected_output']):.2f}")
        print("-" * 50)

def main():
    import sys
    
    # If no arguments provided, run test cases
    if len(sys.argv) == 1:
        run_test_cases()
        return
    
    # If arguments provided, calculate reimbursement
    if len(sys.argv) != 4:
        print("Usage: python lame.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        print("   or: python lame.py (to run test cases)")
        sys.exit(1)
    
    try:
        days = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        if days < 0 or miles < 0 or receipts < 0:
            print("Error: All values must be non-negative")
            sys.exit(1)
        
        result = calculate_reimbursement_v2(days, miles, receipts)
        print(f"{result:.2f}")
        
    except ValueError:
        print("Error: All arguments must be valid numbers")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
