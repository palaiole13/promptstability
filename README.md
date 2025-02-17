# promptstability

[![PyPI](https://img.shields.io/pypi/v/promptstability.svg)](https://pypi.org/project/promptstability/)
[![Tests](https://github.com/palaiole13/promptstability/actions/workflows/test.yml/badge.svg)](https://github.com/palaiole13/promptstability/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/palaiole13/promptstability?include_prereleases&label=changelog)](https://github.com/palaiole13/promptstability/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/palaiole13/promptstability/blob/main/LICENSE)

Package for generating Prompt Stability Scores (PSS). See paper [here](https://www.arxiv.org/abs/2407.02039) outlining technique for investigating the stability of outcomes resulting from variations in language model prompt specifications.

## Requirements

- **Python 3.8 to 3.10** (Python 3.11 and above are not supported due to dependency limitations)
- Other dependencies are installed automatically via `pip`

## Installation

Install this library using `pip`:
```bash
pip install promptstability
```
## Usage

#### OpenAI example
``` python
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
client = OpenAI(api_key=get_openai_api_key()) #Will get an error if no API key set as environment variable

# Enter in terminal: export OPENAI_API_KEY='your-api-key-here'
# OR (not advised) hard code it with:

os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Initialize OpenAI client
client = OpenAI(api_key=get_openai_api_key())

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
```

#### Ollama annotation function example
``` python
import ollama
MODEL = 'llama3'
def annotate(text, prompt, temperature=0.1):
    response = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": f"'{prompt}'"},
        {"role": "user", "content": f"'{text}'"}
    ])
return response['message']['content']
```

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:
```bash
cd promptstability
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
pytest
```
