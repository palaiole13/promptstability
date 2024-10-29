import os
import pandas as pd
from promptstability.core import PromptStabilityAnalysis

# Sample test data
test_data = [
    "Healthcare is a significant political issue.",
    "Environmental policies impact economic growth.",
    "Social reforms improve quality of life."
]

# Mock annotation function for testing
def mock_annotation_function(text, prompt, temperature=0.1):
    """Mock function that returns 1 for any prompt, mimicking a consistent annotation."""
    return 1

# Test function for `intra_pss`
def test_intra_pss():
    psa = PromptStabilityAnalysis(annotation_function=mock_annotation_function, data=test_data)
    
    # Define parameters for intra_pss
    original_text = "This is a prompt about healthcare."
    prompt_postfix = "Answer 1 if related; 0 if not related."
    iterations = 5
    bootstrap_samples = 10  # Use a small number for testing

    # Run intra_pss
    ka_scores, annotated_data = psa.intra_pss(
        original_text=original_text,
        prompt_postfix=prompt_postfix,
        iterations=iterations,
        bootstrap_samples=bootstrap_samples,
        plot=False
    )
    
    # Assertions to verify functionality
    assert isinstance(ka_scores, dict), "KA scores should be a dictionary"
    assert not annotated_data.empty, "Annotated data should not be empty"
    assert "annotation" in annotated_data.columns, "Annotated data should contain 'annotation' column"
