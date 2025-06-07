import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set pandas display options to show all rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Read the JSON file
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([case['input'] for case in data])

# Add expected output to the DataFrame
df['expected_output'] = [case['expected_output'] for case in data]

# Print basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Create a figure with subplots for each parameter
plt.figure(figsize=(15, 10))

# Plot 1: Trip Duration Days
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='trip_duration_days', bins=5)
plt.title('Distribution of Trip Duration Days')
plt.xlabel('Days')

# Plot 2: Miles Traveled
plt.subplot(2, 2, 2)
sns.histplot(data=df, x='miles_traveled', bins=20)
plt.title('Distribution of Miles Traveled')
plt.xlabel('Miles')

# Plot 3: Total Receipts Amount
plt.subplot(2, 2, 3)
sns.histplot(data=df, x='total_receipts_amount', bins=20)
plt.title('Distribution of Total Receipts Amount')
plt.xlabel('Amount ($)')

# Plot 4: Expected Output
plt.subplot(2, 2, 4)
sns.histplot(data=df, x='expected_output', bins=20)
plt.title('Distribution of Expected Output')
plt.xlabel('Amount ($)')

plt.tight_layout()
plt.savefig('parameter_distributions.png')
plt.close()

# Print correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Print value counts for trip_duration_days
print("\nTrip Duration Days Value Counts:")
print(df['trip_duration_days'].value_counts().sort_index())

# Print sorted miles traveled and corresponding expected outputs
print("\nSorted Miles Traveled vs Expected Output:")
sorted_df = df.sort_values('expected_output')
print(pd.DataFrame({
    'Miles Traveled': sorted_df['miles_traveled'],
    'Expected Output': sorted_df['expected_output'],
    'Trip Duration': sorted_df['trip_duration_days'],
    'Receipts Amount': sorted_df['total_receipts_amount']
}))
