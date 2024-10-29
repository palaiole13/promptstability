import pytest
from unittest.mock import MagicMock
import pandas as pd

from promptstability.core import PromptStabilityAnalysis

@pytest.fixture
def setup_analysis():
    # Mock data and annotation function
    data = pd.DataFrame({
        'text': ['sample text'] * 10,  # Example data repeated 10 times
        'id': list(range(10))
    })

    # Example function that just returns a mock annotation
    def mock_annotation_function(text, prompt):
        return "annotated_" + text

    analysis = PromptStabilityAnalysis(annotation_function=mock_annotation_function, data=data, metric_fn=lambda x: x)
    return analysis

def test_intra_pss(setup_analysis):
    analysis = setup_analysis
    original_text = "Test text"
    prompt_postfix = "Test postfix"
    iterations = 2  # Reduce for testing purposes
    bootstrap_samples = 10  # Reduce for testing purposes

    # Run the baseline_stochasticity method
    ka_scores, annotated_data = analysis.intra_pss(original_text, prompt_postfix, iterations, bootstrap_samples)

    # Assertions
    assert isinstance(ka_scores, dict), "KA scores should be a dictionary"
    assert isinstance(annotated_data, pd.DataFrame), "Annotated data should be a DataFrame"
    assert 'ka_mean' in annotated_data.columns, "KA mean should be calculated and added to the DataFrame"

    # Check if KA scores are calculated for each iteration starting from 1
    assert all(i in ka_scores for i in range(1, iterations)), "KA scores should be calculated for each iteration"
    assert all(isinstance(ka_scores[i], dict) for i in range(1, iterations)), "Each entry in ka_scores should be a dictionary"

    print("Test passed: KA calculations and DataFrame outputs are correct.")
