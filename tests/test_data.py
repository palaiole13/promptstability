import pandas as pd
from promptstability import load_example_data  # Adjust if load_example_data is in a different module

try:
    # Attempt to load the example data
    df = load_example_data()
    print("Example data loaded successfully:")
    print(df.head())
except FileNotFoundError:
    print("Error: example_data.csv not found in the package.")
except Exception as e:
    print(f"An error occurred: {e}")