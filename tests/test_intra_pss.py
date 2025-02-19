import os
import pandas as pd
from promptstability.core import PromptStabilityAnalysis, get_openai_api_key
from openai import OpenAI
import pytest

# Initialize OpenAI client
APIKEY = get_openai_api_key()
client = OpenAI(api_key=APIKEY)

# Define the annotation function with the specified API call structure
def annotation_function(text, prompt, temperature=0.1):
    """Annotation function using OpenAI API client for chat completions."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        return ''.join(choice.message.content for choice in response.choices)
    except Exception as e:
        print(f"Caught exception: {e}")
        return None

# Sample test data
test_data = [
    "The quick brown fox jumps over the lazy dog.",
    "The economic policies are boosting the economy.",
    "Climate change is affecting our planet.",
    "Immigration policies are being reformed.",
    "Healthcare is a critical issue for many people."
]

# Initialize PromptStabilityAnalysis with the test annotation function and data
psa = PromptStabilityAnalysis(annotation_function=annotation_function, data=test_data)

@pytest.mark.requires_api_key
def test_intra_pss():
    """Test the intra_pss function (within-prompt stability) with iterative CSV output."""
    original_text = "This is a prompt about the quick brown fox."
    prompt_postfix = "Answer 1 if related; 0 if not related."
    iterations = 20
    bootstrap_samples = 50  # Use a lower value for faster testing

    all_annotations = []  # Collect all annotations here for CSV writing

    # Step 1: Run all iterations to collect annotations
    for i in range(iterations):
        print(f"Running iteration {i+1} of {iterations}...")

        # Annotate each text item without KA calculation
        for j, text in enumerate(test_data):
            annotation = annotation_function(text, f"{original_text} {prompt_postfix}")
            all_annotations.append({'id': j, 'text': text, 'annotation': annotation, 'iteration': i})

    # Step 2: Save all annotations to CSV for inspection
    annotated_data = pd.DataFrame(all_annotations)
    annotated_data.to_csv("tests/annotations/test_intra_annotations.csv", index=False)
    print("All annotated data saved to test_intra_annotations.csv")

    # Step 3: Perform KA calculation on the full dataset, if desired
    try:
        ka_scores, _ = psa.bootstrap_krippendorff(annotated_data, annotator_col='iteration', bootstrap_samples=bootstrap_samples)
        print("\nKrippendorff's Alpha Scores:")
        print(ka_scores)
    except ZeroDivisionError:
        print("ZeroDivisionError encountered in Krippendorff's Alpha calculation. Check test_annotations.csv for annotation details.")
    except Exception as e:
        print(f"An error occurred during KA calculation: {e}")

# Run the tests if this file is executed directly
if __name__ == "__main__":
    print("Running test for intra_pss (within-prompt stability) with iterative CSV output...")
    test_intra_pss()
