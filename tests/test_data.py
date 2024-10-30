import pandas as pd
from promptstability import load_example_data

def test_load_example_data():
    try:
        df = load_example_data()
        assert not df.empty, "DataFrame is empty"
        print("Example data loaded successfully:")
        print(df.head())
    except FileNotFoundError:
        pytest.fail("Error: example_data.csv not found in the package.")
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")
