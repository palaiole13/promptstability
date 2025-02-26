# promptstability

[![PyPI](https://img.shields.io/pypi/v/promptstability.svg)](https://pypi.org/project/promptstability/)
[![Tests](https://github.com/palaiole13/promptstability/actions/workflows/test.yml/badge.svg)](https://github.com/palaiole13/promptstability/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/palaiole13/promptstability?include_prereleases&label=changelog)](https://github.com/palaiole13/promptstability/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/palaiole13/promptstability/blob/main/LICENSE)

Package for generating Prompt Stability Scores (PSS). See paper [here](https://www.arxiv.org/abs/2407.02039) outlining technique for investigating the stability of outcomes resulting from variations in language model prompt specifications. Replication material [here](https://github.com/cjbarrie/promptstability/tree/main).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Example Usage](#example-usage)
- [API Documentation](#api-documentation)
- [Development](#development)

## Requirements

- **Python 3.8 to 3.10** (Python 3.11 and above are not supported due to dependency limitations)
- Other dependencies are installed automatically via `pip`

## Installation

Install this library using `pip`:
```bash
pip install promptstability
```
## Example Usage
Here we provide instructions for using `promptstability` with OpenAI and Ollama.

``` python
import pandas as pd
from promptstability.core import get_api_key
from promptstability.core import PromptStabilityAnalysis
from promptstability.core import load_example_data
import os

# Load data (news articles)
df = load_example_data()
print(df.head())
example_data = list(df['body'].values) # Take a subsample

# Define the prompt texts
original_text = 'The following are some news articles about the economy.'
prompt_postfix = 'Respond 0 for positive news, or 1 for negative news. Guess if you do not know. Respond nothing else.'
```
#### a) OpenAI Example (e.g., GPT-4o-mini)
```python
from openai import OpenAI

# Initialize OpenAI client
# First set the OPENAI_API_KEY environment variable
APIKEY = get_api_key('openai')
client = OpenAI(api_key=APIKEY)

OPENAI_MODEL = 'gpt-4o-mini'

# Define the OpenAI annotation function
def annotate_openai(text, prompt, temperature=0.1):
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
    except Exception as e:
        print(f"OpenAI exception: {e}")
        raise e

    return ''.join(choice.message.content for choice in response.choices)

# Instantiate the analysis class using OpenAI’s annotation function (Note on warnings: Pegasus comes with automated warning about model weights, which you can ignore)
psa_openai = PromptStabilityAnalysis(annotation_function=annotate_openai, data=example_data)

# Run intra-prompt stability analysis using the method `intra_pss`
print("Running OpenAI intra-prompt analysis...")
ka_openai_intra, annotated_openai_intra = psa_openai.intra_pss(
    original_text,
    prompt_postfix,
    iterations=5,   # minimal iterations
    plot=True,
    save_path='news_intra.png',
    save_csv="news_intra.csv"
)
print("OpenAI intra-prompt KA scores:", ka_openai_intra)

# Run inter-prompt stability analysis using the method `inter_pss`
print("Running OpenAI inter-prompt analysis...")
temperatures = [0.1, 0.5, 2.0] # in practice, you would set more temperatures than this
ka_openai_inter, annotated_openai_inter = psa_openai.inter_pss(
    original_text,
    prompt_postfix,
    nr_variations=3,
    temperatures=temperatures,
    iterations=1,
    plot=True,
    save_path='news_inter.png',
    save_csv="news_inter.csv"
)
print("OpenAI inter-prompt KA scores:", ka_openai_inter)
```

#### b) Ollama Example (e.g., your local deepseek-r1:8b)
``` python
import ollama

# Make sure that your Ollama server is running locally and that 'deepseek-r1:8b' is available.
OLLAMA_MODEL = 'deepseek-r1:8b'

# Define the Ollama annotation function
def annotate_ollama(text, prompt, temperature=0.1):
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ])
    except Exception as e:
        print(f"Ollama exception: {e}")
        raise e
    return response['message']['content']

# Instantiate the analysis class using Ollama’s annotation function (Note on warnings: Pegasus comes with automated warning about model weights, which you can ignore)
psa_ollama = PromptStabilityAnalysis(annotation_function=annotate_ollama, data=example_data)

# Run intra-prompt stability analysis using the method `intra_pss`
print("Running Ollama intra-prompt analysis...")
ka_ollama_intra, annotated_ollama_intra = psa_ollama.intra_pss(
    original_text,
    prompt_postfix,
    iterations=5,
    plot=False
)
print("Ollama intra-prompt KA scores:", ka_ollama_intra)

# Run inter-prompt stability analysis using the method `inter_pss`
temperatures = [0.1, 2.0, 5.0]  # or whichever temperatures you want to test
print("Running Ollama inter-prompt analysis...")
ka_ollama_inter, annotated_ollama_inter = psa_ollama.inter_pss(
    original_text,
    prompt_postfix,
    nr_variations=3,
    temperatures=temperatures,
    iterations=1,
    plot=False
)
print("Ollama inter-prompt KA scores:", ka_ollama_inter)
```
## API Documentation
Our full API reference documentation is hosted on Read the Docs and includes detailed information on all modules, classes, and functions.

You can access the documentation here:

[PromptStability API Documentation](https://promptstability.readthedocs.io)

*This documentation is automatically updated whenever changes are pushed to the repository.*

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
