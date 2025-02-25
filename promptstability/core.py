import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import simpledorff
import pandas as pd
import numpy as np
import time
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import importlib.resources

def load_example_data():
    """Loads example data included with the package."""
    with importlib.resources.open_text("promptstability.data", "example_data.csv") as f:
        return pd.read_csv(f)

def get_api_key(api: str = "openai") -> str:
    """
    Retrieve the API key for the specified service from environment variables.

    Parameters
    ----------
    api : str, optional
        The name of the API service (e.g., "openai", "mistral"). (default: "openai").

    Returns
    -------
    str
        The API key retrieved from the corresponding environment variable.

    Raises
    ------
    ValueError
        If the API key is not found or if the API service is unsupported.
    """
    env_var_map = {
    "openai": "OPENAI_API_KEY",           # For OpenAI's GPT models (e.g., GPT-3.5, GPT-4)
    "mistral": "MISTRAL_API_KEY",         # For Mistral's models
    "anthropic": "ANTHROPIC_API_KEY",     # For Anthropic's Claude models
    "cohere": "COHERE_API_KEY",           # For Cohere's language models
    "huggingface": "HUGGINGFACE_API_KEY" # For accessing models via Hugging Face's Inference API (sometimes called HUGGINGFACE_HUB_TOKEN)
}

    key_name = env_var_map.get(api.lower())
    if not key_name:
        raise ValueError(f"Unsupported API: {api}. Supported APIs: {list(env_var_map.keys())}")

    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"API key not found. Please set the {key_name} environment variable.")

    return api_key

class PromptStabilityAnalysis:

    def __init__(self, annotation_function, data, metric_fn=simpledorff.metrics.nominal_metric, parse_function=None) -> None:
        """
        Initialize a PromptStabilityAnalysis instance.

        Sets up models and functions for generating paraphrases and evaluating intra- and inter-prompt stability
        via annotation agreement (Krippendorff's Alpha).

        Parameters
        ----------
        annotation_function : callable
            A function that takes a text and a prompt and returns an annotation.
        data : list
            A list of text items to be annotated.
        metric_fn : callable, optional
            Function to compute the agreement metric (default: simpledorff.metrics.nominal_metric).
            The default nominal metric (metric_fn=simpledorff.metrics.nominal_metric) should be used when annotations are categorical, with no inherent ordering.
            If annotations are ordinal or numerical, the alternative interval metric should be used (metric_fn=simpledorff.metrics.interval_metric).
        parse_function : callable, optional
            Function to post-process the raw output of the annotation_function. If None, the default raw output is used (default: lambda x: x).
        """
        self.annotation_function = annotation_function
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        model_name = 'tuner007/pegasus_paraphrase'
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.parse_function = parse_function if parse_function is not None else lambda x: x  # Default parse function
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)
        self.data = data
        self.metric_fn = metric_fn

    def __paraphrase_sentence(self, input_text, num_return_sequences=10, num_beams=50, temperature=1.0):
        batch = self.tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(self.torch_device)
        translated = self.model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temperature, do_sample=True)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def __generate_paraphrases(self, original_text, prompt_postfix, nr_variations, temperature=1.0):
        phrases = self.__paraphrase_sentence(original_text, num_return_sequences=nr_variations, temperature=temperature)
        l = [{'phrase': f'{original_text} {prompt_postfix}', 'original': True}]
        for phrase in phrases:
            l.append({'phrase': f'{phrase} {prompt_postfix}', 'original': False})
        self.paraphrases = pd.DataFrame(l)
        return self.paraphrases

    def intra_pss(self, original_text, prompt_postfix=None, iterations=10, bootstrap_samples=1000, plot=False, save_path=None, save_csv=None):
        """
        Evaluate prompt stability within a single prompt context via repeated annotation (intra-PSS).

        This method takes the original prompt (original_text plus prompt_postfix) and repeatedly classifies the same n rows of data.
        For each iteration, this method then calculates a cumulative reliability score (Krippendorff’s Alpha (KA)), using
        bootstrapping to estimate agreement and its confidence intervals. This is the intra-prompt stability score or intra-PSS.

        Parameters
        ----------
        original_text : str
            The base text of the prompt.
        prompt_postfix : str, optional
            Additional text appended to specify the type of output required (e.g., binary, interval). (default: None).
        iterations : int, optional
            Number of annotation iterations (default: 10).
        bootstrap_samples : int, optional
            Number of bootstrap samples for estimating confidence intervals (default: 1000).
        plot : bool, optional
            If True, plot the KA scores (default: False).
        save_path : str, optional
            File path to save the plot (default: None).
            Parameter plot should be set to True if plot is to be saved.
        save_csv : str, optional
            File path to save the annotated data as a CSV file (default: None).

        Returns
        -------
        tuple
            A tuple containing:
            - A dictionary of cumulative KA scores per iteration.
            - A pandas.DataFrame of the annotated data and corresponding KA metrics.
        """
        prompt = f'{original_text} {prompt_postfix}'
        all_annotations = []  # Use a list to collect all annotations

        ka_scores = {}

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...", end='\r')
            sys.stdout.flush()

            annotations = []

            for j, d in enumerate(self.data):
                annotation = self.parse_function(self.annotation_function(d, prompt))
                annotations.append({'id': j, 'text': d, 'annotation': annotation, 'iteration': i})

            all_annotations.extend(annotations)  # Extend the list with the current iteration's annotations

        all_annotated = pd.DataFrame(all_annotations)  # Convert list to DataFrame once

        # Calculate Krippendorff's Alpha after collecting >1 set of annotations
        for i in range(1, iterations):
            annotator_col = 'iteration'
            mean_alpha, (ci_lower, ci_upper) = self.bootstrap_krippendorff(all_annotated[all_annotated['iteration'] <= i], annotator_col, bootstrap_samples)
            ka_scores[i] = {'Average Alpha': mean_alpha, 'CI Lower': ci_lower, 'CI Upper': ci_upper}

        # Adding average KA, CI lower, and CI upper to the combined data for CSV output
        for i in ka_scores:
            all_annotated.loc[all_annotated['iteration'] == i, 'ka_mean'] = ka_scores[i]['Average Alpha']
            all_annotated.loc[all_annotated['iteration'] == i, 'ka_lower'] = ka_scores[i]['CI Lower']
            all_annotated.loc[all_annotated['iteration'] == i, 'ka_upper'] = ka_scores[i]['CI Upper']

        if save_csv:
            all_annotated.to_csv(save_csv, index=False)
            print(f"Annotated data saved to {save_csv}")

        if plot:
            # Function to plot KA scores with integer x-axis labels
            iterations_list = list(ka_scores.keys())
            ka_values = [ka_scores[i]['Average Alpha'] for i in iterations_list]
            average_ka = np.mean(ka_values)
            ci_lowers = [ka_scores[i]['Average Alpha'] - ka_scores[i]['CI Lower'] for i in iterations_list]
            ci_uppers = [ka_scores[i]['CI Upper'] - ka_scores[i]['Average Alpha'] for i in iterations_list]

            plt.figure(figsize=(10, 5))
            plt.errorbar(iterations_list, ka_values, yerr=[ci_lowers, ci_uppers], fmt='o', linestyle='-', color='b', ecolor='gray', capsize=3)
            plt.axhline(y=average_ka, color='r', linestyle='--', label=f'Average KA: {average_ka:.2f}')
            plt.xlabel('Iteration')
            plt.ylabel("Krippendorff's Alpha (KA)")
            plt.title("Krippendorff's Alpha Scores with 95% CI Across Iterations")
            plt.xticks(iterations_list)  # Set x-axis ticks to be whole integers
            plt.legend()
            plt.grid(True)
            plt.axhline(y=0.8, color='black', linestyle='--', linewidth=.5)

            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

        return ka_scores, all_annotated

    def inter_pss(self, original_text, prompt_postfix=None, nr_variations=5, temperatures=[0.1, 2.5, 5.0], iterations=1, bootstrap_samples=1000, print_prompts=False, edit_prompts_path=None, plot=False, save_path=None, save_csv=None):
        """
        Evaluate prompt stability across semantically similar prompts (inter-PSS).

        This method uses PEGASUS to generate multiple paraphrases of the original prompt,
        and the temperature feature to control the degree of similarity between these prompts
        (the higher the temperature, the larger the semantic distance between the original prompts and the genarated paraphrases).
        For each temperature setting, this method then computes Krippendorff's Alpha (with bootstrapping) to assess
        annotation agreement. This is the inter-prompt stability score or inter-PSS.

        Parameters
        ----------
        original_text : str
            The base text of the prompt.
        prompt_postfix : str, optional
            Additional text appended to specify the type of output required (e.g., binary, interval).
            This string  does not get paraphrased - i.e. remains constant across all variations (default: None).
        nr_variations : int, optional
            Number of paraphrase variations to generate per temperature (default: 5).
        temperatures : list, optional
            A list of temperature values for paraphrase generation (default: [0.1, 2.5, 5.0]).
        iterations : int, optional
            Number of annotation iterations per temperature (default: 1).
        bootstrap_samples : int, optional
            Number of bootstrap samples for estimating Krippendorff's Alpha (default: 1000).
        print_prompts : bool, optional
            If True, print the unique generated prompts (default: False).
        edit_prompts_path : str, optional
            File path to save the prompts for manual editing, to be used in the manual_inter_pss method (default: None).
        plot : bool, optional
            If True, plot the KA scores (default: False).
        save_path : str, optional
            File path to save the plot (default: None).
        save_csv : str, optional
            File path to save the annotated data as CSV (default: None).

        Returns
        -------
        tuple
            A tuple containing:
            - A dictionary of KA scores for each temperature.
            - A combined pandas.DataFrame of all annotated data.
        """
        ka_scores = {}
        all_annotated = []

        for temp in temperatures:
            paraphrases = self.__generate_paraphrases(original_text, prompt_postfix, nr_variations=nr_variations, temperature=temp)
            annotated = []

            for i in range(iterations):
                start_time = time.time() #
                for j, (paraphrase, original) in enumerate(zip(paraphrases['phrase'], paraphrases['original'])):

                    print(f"Temperature {temp}, Iteration {i+1}/{iterations}", end='\r')
                    sys.stdout.flush()
                    for k, d in enumerate(self.data):
                        annotation = self.parse_function(self.annotation_function(d, paraphrase))
                        annotated.append({'id': k, 'text': d, 'annotation': annotation, 'prompt_id': j, 'prompt': paraphrase, 'original': original, 'temperature': temp})

                end_time = time.time()  #
                elapsed_time = end_time - start_time  #
                print(f"Temperature {temp} completed in {elapsed_time:.2f} seconds")

            annotated_data = pd.DataFrame(annotated)
            all_annotated.append(annotated_data)

            # Bootstrap Krippendorff's Alpha calculation for each temperature
            annotator_col = 'prompt_id'
            print(f'KA calculation for {bootstrap_samples} bootstrap samples...')
            mean_alpha, (ci_lower, ci_upper) = self.bootstrap_krippendorff(annotated_data, annotator_col, bootstrap_samples)
            ka_scores[temp] = {'Average Alpha': mean_alpha, 'CI Lower': ci_lower, 'CI Upper': ci_upper}
            print(f'KA calculation completed.')
            print()

        # Concatenate all annotated data
        combined_annotated_data = pd.concat(all_annotated, ignore_index=True)

         # Add average KA, CI lower, and CI upper to the combined data for CSV output
        for temp in ka_scores:
            combined_annotated_data.loc[combined_annotated_data['temperature'] == temp, 'ka_mean'] = ka_scores[temp]['Average Alpha']
            combined_annotated_data.loc[combined_annotated_data['temperature'] == temp, 'ka_lower'] = ka_scores[temp]['CI Lower']
            combined_annotated_data.loc[combined_annotated_data['temperature'] == temp, 'ka_upper'] = ka_scores[temp]['CI Upper']

        if save_csv:
            combined_annotated_data.to_csv(save_csv, index=False)
            print(f"Annotated data saved to {save_csv}")

        if print_prompts:
            unique_prompts = combined_annotated_data['prompt'].unique()
            print("Unique prompts:")
            for prompt in unique_prompts:
                print(prompt)

        if edit_prompts_path:
            prompts_df = combined_annotated_data.drop_duplicates(subset=['prompt_id', 'temperature', 'prompt', 'original'])
            prompts_df = prompts_df[['prompt_id', 'temperature', 'prompt', 'original']]
            prompts_df.columns = ['prompt_id', 'temperature', 'prompt_text', 'original_prompt']
            prompts_df.to_csv(edit_prompts_path, index=False)
            print(f"{nr_variations} prompts per temperature saved and available to edit at {edit_prompts_path}")

        if plot:
            temperatures_list = list(ka_scores.keys())
            ka_values = [ka_scores[temp]['Average Alpha'] for temp in temperatures_list]
            ka_lowers = [ka_scores[temp]['Average Alpha'] - ka_scores[temp]['CI Lower'] for temp in temperatures_list]
            ka_uppers = [ka_scores[temp]['CI Upper'] - ka_scores[temp]['Average Alpha'] for temp in temperatures_list]

            plt.figure(figsize=(10, 5))
            plt.plot(temperatures_list, ka_values, marker='o', linestyle='-', color='b')
            plt.errorbar(temperatures_list, ka_values, yerr=[ka_lowers, ka_uppers], fmt='o', linestyle='-', color='b', ecolor='gray', capsize=3)
            plt.xlabel('Temperature')
            plt.ylabel('Krippendorff\'s Alpha (KA)')
            plt.title('Krippendorff\'s Alpha Scores with 95% CI Across Temperatures')
            plt.xticks(temperatures_list)  # Set x-axis ticks to be whole integers
            plt.grid(True)
            plt.ylim(0.0, 1.05)
            plt.axhline(y=0.80, color='black', linestyle='--', linewidth=.5)

            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

        return ka_scores, combined_annotated_data

    def manual_inter_pss(self, edit_prompts_path, bootstrap_samples=1000, plot=False, save_path=None, save_csv=None):
        """
        Evaluate prompt stability across semantically similar prompts (inter-PSS).
        The only difference from inter_pss is that prompts here are manually edited (rather than automatically generated).

        Parameters
        ----------
        edit_prompts_path : str
            Path to the CSV file containing the manually edited prompts. This could be the same as the edit_prompts_path in the inter_pss method.
        bootstrap_samples : int, optional
            Number of bootstrap samples to use for estimating Krippendorff's Alpha (default is 1000).
        plot : bool, optional
            If True, plots the Krippendorff's Alpha scores (default is False).
        save_path : str, optional
            File path to save the plot (default is None).
        save_csv : str, optional
            File path to save the combined annotated data as CSV (default is None).

        Returns
        -------
        tuple
            A tuple containing:
            - A dictionary of Krippendorff's Alpha scores for each temperature.
            - A pandas.DataFrame of the combined annotated data.
        """
        # Load the manually edited prompts CSV
        prompts_df = pd.read_csv(edit_prompts_path)

        # Assuming 'original_prompt' column is used to filter out original, unedited prompts
        prompts_df = prompts_df[prompts_df['original_prompt'] == False]

        ka_scores = {}
        all_annotated = []

        # Iterate through each unique temperature found in the prompts DataFrame
        for temp in prompts_df['temperature'].unique():
            temp_prompts = prompts_df[prompts_df['temperature'] == temp]
            annotated = []
            start_time = time.time()

            # Annotate data using each prompt at the current temperature
            for _, prompt_entry in temp_prompts.iterrows():
                prompt = prompt_entry['prompt_text']
                prompt_id = prompt_entry['prompt_id']

                for k, d in enumerate(self.data):
                    annotation = self.parse_function(self.annotation_function(d, prompt))
                    annotated.append({
                        'id': k,
                        'text': d,
                        'annotation': annotation,
                        'prompt_id': prompt_id,
                        'prompt': prompt,
                        'temperature': temp
                    })

            end_time = time.time()  #
            elapsed_time = end_time - start_time  #
            print(f"Temperature {temp} completed in {elapsed_time:.2f} seconds")

            annotated_data = pd.DataFrame(annotated)
            all_annotated.append(annotated_data)

            # Bootstrap Krippendorff's Alpha calculation for each temperature
            print(f'KA calculation for {bootstrap_samples} bootstrap samples...')
            mean_alpha, (ci_lower, ci_upper) = self.bootstrap_krippendorff(annotated_data, 'prompt_id', bootstrap_samples)
            ka_scores[temp] = {'Average Alpha': mean_alpha, 'CI Lower': ci_lower, 'CI Upper': ci_upper}
            print(f'KA calculation completed.')
            print()

        # Concatenate all annotated data
        combined_annotated_data = pd.concat(all_annotated, ignore_index=True)

        # Add average KA, CI lower, and CI upper to the combined data for CSV output
        for temp in ka_scores:
            combined_annotated_data.loc[combined_annotated_data['temperature'] == temp, 'ka_mean'] = ka_scores[temp]['Average Alpha']
            combined_annotated_data.loc[combined_annotated_data['temperature'] == temp, 'ka_lower'] = ka_scores[temp]['CI Lower']
            combined_annotated_data.loc[combined_annotated_data['temperature'] == temp, 'ka_upper'] = ka_scores[temp]['CI Upper']

        # Output results as needed
        if save_csv:
            combined_annotated_data.to_csv(save_csv, index=False)
            print(f"Annotated data saved to {save_csv}")

        if plot:
            temperatures_list = list(ka_scores.keys())
            ka_values = [ka_scores[temp]['Average Alpha'] for temp in temperatures_list]
            ka_lowers = [ka_scores[temp]['Average Alpha'] - ka_scores[temp]['CI Lower'] for temp in temperatures_list]
            ka_uppers = [ka_scores[temp]['CI Upper'] - ka_scores[temp]['Average Alpha'] for temp in temperatures_list]

            plt.figure(figsize=(10, 5))
            plt.plot(temperatures_list, ka_values, marker='o', linestyle='-', color='b')
            plt.errorbar(temperatures_list, ka_values, yerr=[ka_lowers, ka_uppers], fmt='o', linestyle='-', color='b', ecolor='gray', capsize=3)
            plt.xlabel('Temperature')
            plt.ylabel('Krippendorff\'s Alpha (KA)')
            plt.title('Krippendorff\'s Alpha Scores with 95% CI Across Temperatures')
            plt.xticks(temperatures_list)  # Set x-axis ticks to be whole integers
            plt.grid(True)
            plt.ylim(0.0, 1.05)
            plt.axhline(y=0.80, color='black', linestyle='--', linewidth=.5)

            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

        return ka_scores, combined_annotated_data

    def bootstrap_krippendorff(self, df, annotator_col, bootstrap_samples, confidence_level=95):
        """
        Compute Krippendorff's Alpha using bootstrapping.

        This method resamples the provided DataFrame with replacement to generate a distribution
        of Krippendorff's Alpha values, from which the mean and confidence intervals are computed.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing annotation data.
        annotator_col : str
            Name of the column identifying the annotator (or iteration) in the DataFrame.
        bootstrap_samples : int
            Number of bootstrap resampling iterations.
        confidence_level : int, optional
            Confidence level for the computed interval (default: 95).

        Returns
        -------
        tuple
            A tuple (mean_alpha, (ci_lower, ci_upper)) where:
            - mean_alpha is the mean Krippendorff's Alpha over bootstrap samples.
            - ci_lower and ci_upper are the lower and upper bounds of the confidence interval.
        """
        alpha_scores = []

        for _ in range(bootstrap_samples):  # Number of bootstrap samples
            bootstrap_sample = df.sample(n=len(df), replace=True)

            alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
                bootstrap_sample,
                metric_fn=self.metric_fn,
                experiment_col='id',
                annotator_col=annotator_col,
                class_col='annotation'
            )
            alpha_scores.append(alpha)

        alpha_scores = np.array(alpha_scores)
        mean_alpha = np.mean(alpha_scores)
        ci_lower = np.percentile(alpha_scores, (100 - confidence_level) / 2)
        ci_upper = np.percentile(alpha_scores, 100 - (100 - confidence_level) / 2)
        return mean_alpha, (ci_lower, ci_upper)
