#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Check if all three parameters are provided
if [ $# -ne 3 ]; then
    echo "Error: This script requires exactly 3 parameters"
    echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    exit 1
fi

# Run the Python implementation
python3 infer_eval.py "$1" "$2" "$3"