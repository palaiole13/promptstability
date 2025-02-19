from openai import OpenAI
import pandas as pd
from promptstability.core import get_openai_api_key
from promptstability.core import PromptStabilityAnalysis
from promptstability.core import load_example_data

import os
print("OPENAI_API_KEY:", os.environ.get('OPENAI_API_KEY'))

#This script mimics a user run-through of package use

# Load data
df = load_example_data()
print(df.head())

# Take a subsample
example_data = list(df['body'].values)

# Initialize OpenAI client
APIKEY = get_openai_api_key()
client = OpenAI(api_key=APIKEY)

# Define the annotation function
def annotate(text, prompt, temperature=0.1):
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
    except Exception as e:
        print(f"Caught exception: {e}")
        raise e

    return ''.join(choice.message.content for choice in response.choices)

psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data)

# Construct the prompt
original_text = 'The following are some news articles about the economy.'
prompt_postfix = '[Respond 0 for positive news, or 1 for negative news. Guess if you do not know. Respond nothing else.]'

# Run intra_pss (aka within-prompt PSS)
ka_scores, annotated_data = psa.intra_pss(original_text, prompt_postfix, iterations=20, plot=True, save_path='news_within.png', save_csv="news_within.csv")

# Run inter_pss (aka between-prompt PSS)
# Set temperatures (in practice, you would set more temperatures than this)
temperatures = [0.1, 5.0]

# Get KA scores across different temperature paraphrasings
ka_scores, annotated_data = psa.inter_pss(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 1, print_prompts=True, plot=True, save_path='news_between.png', save_csv = 'news_between.csv')
