import os
import pandas as pd
from promptstability.core import PromptStabilityAnalysis, get_openai_api_key
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=get_openai_api_key())

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

def test_inter_pss():
    """Test the inter_pss function (between-prompt stability) with iterative CSV output."""
    original_text = "This is a political statement about healthcare."
    prompt_postfix = "Answer 1 if related; 0 if not related."
    temperatures = [0.1, 0.5, 1.0]  # Adjust as needed
    nr_variations = 5
    iterations = 3  # Use a lower value for faster testing
    bootstrap_samples = 50  # Use a lower value for faster testing

    all_annotations = []  # Collect all annotations here for CSV writing

    # Step 1: Run all temperatures and iterations to collect annotations
    for temp in temperatures:
        print(f"Running annotations for temperature {temp}...")
        
        for i in range(iterations):
            print(f"Iteration {i+1} of {iterations} for temperature {temp}...")
            
            # Generate paraphrased prompts for each temperature
            paraphrases = psa._PromptStabilityAnalysis__generate_paraphrases(original_text, prompt_postfix, nr_variations, temperature=temp)
            
            # Annotate each paraphrased prompt without KA calculation
            for j, row in paraphrases.iterrows():
                paraphrased_prompt = row['phrase']
                
                for k, text in enumerate(test_data):
                    annotation = annotation_function(text, paraphrased_prompt, temperature=temp)
                    all_annotations.append({
                        'id': k,
                        'text': text,
                        'annotation': annotation,
                        'temperature': temp,
                        'prompt_id': j,
                        'iteration': i,
                        'paraphrased_prompt': paraphrased_prompt
                    })

    # Step 2: Save all annotations to CSV for inspection
    annotated_data = pd.DataFrame(all_annotations)
    annotated_data.to_csv("tests/annotations/test_inter_annotations.csv", index=False)
    print("All annotated data saved to test_inter_annotations.csv")

    # Step 3: Perform KA calculation on the full dataset, if desired
    try:
        ka_scores, _ = psa.bootstrap_krippendorff(annotated_data, annotator_col='prompt_id', bootstrap_samples=bootstrap_samples)
        print("\nKrippendorff's Alpha Scores:")
        print(ka_scores)
    except ZeroDivisionError:
        print("ZeroDivisionError encountered in Krippendorff's Alpha calculation. Check test_inter_annotations.csv for annotation details.")
    except Exception as e:
        print(f"An error occurred during KA calculation: {e}")

# Run the tests if this file is executed directly
if __name__ == "__main__":
    print("Running test for inter_pss (between-prompt stability) with iterative CSV output...")
    test_inter_pss()
