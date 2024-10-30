import pandas as pd
from promptstability.core import PromptStabilityAnalysis

# Sample test data
test_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Economic growth is essential for development.",
    "Climate change impacts biodiversity."
]

# Mock annotation function for testing
def mock_annotation_function(text, prompt, temperature=0.1):
    """Mock function that returns a binary response (1 or 0) based on prompt length for variation."""
    return 1 if len(prompt) % 2 == 0 else 0

# Test function for `inter_pss`
def test_inter_pss():
    psa = PromptStabilityAnalysis(annotation_function=mock_annotation_function, data=test_data)
    
    # Define parameters for inter_pss
    original_text = "This is a political statement."
    prompt_postfix = "Answer 1 if related; 0 if not related."
    temperatures = [0.1, 0.5]
    nr_variations = 3
    iterations = 20
    bootstrap_samples = 10  # For testing purposes

    # Run inter_pss
    ka_scores, annotated_data = psa.inter_pss(
        original_text=original_text,
        prompt_postfix=prompt_postfix,
        nr_variations=nr_variations,
        temperatures=temperatures,
        iterations=iterations,
        plot=False
    )

    # Assertions to verify functionality
    assert isinstance(ka_scores, dict), "KA scores should be a dictionary"
    assert not annotated_data.empty, "Annotated data should not be empty"
    assert "temperature" in annotated_data.columns, "Annotated data should contain 'temperature' column"
