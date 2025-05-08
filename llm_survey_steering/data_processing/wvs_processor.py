# llm_survey_steering/data_processing/wvs_processor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Imports from within the package - assuming config.py is in the parent directory
from ..config import (
    TRAINING_PROMPT_FORMAT, INFERENCE_PROMPT_FORMAT,
    ECON_QUESTIONS_MAP # If used directly here, otherwise pass from config object
)


def map_age_to_group(age):
    """Simple function to categorize age into groups."""
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if age < 30:
        return "Young"
    elif age < 50:
        return "Mid"
    else:
        return "Old"

def load_wvs_data(file_path):
    """Loads WVS data from the specified CSV file path."""
    try:
        wvs_full = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded WVS data. Shape: {wvs_full.shape}")
        return wvs_full
    except FileNotFoundError:
        print(f"Error: WVS data file not found at {file_path}. Please check the path.")
        return None

def preprocess_wvs_data(df, countries, general_vars, demo_vars, econ_q_ids):
    """Subsets and preprocesses the WVS dataframe."""
    columns_to_select = general_vars + demo_vars + econ_q_ids
    # Ensure all selected columns exist in the dataframe
    columns_to_select = [col for col in columns_to_select if col in df.columns]

    if not df['B_COUNTRY_ALPHA'].isin(countries).any():
        print(f"Warning: None of the target countries {countries} found in the WVS data's B_COUNTRY_ALPHA column.")
        # Create an empty DataFrame with expected columns to prevent downstream errors
        # Or handle this more gracefully based on desired behavior
        return pd.DataFrame(columns=columns_to_select + ['Age_Group'])


    subset_df = df[df['B_COUNTRY_ALPHA'].isin(countries)][columns_to_select].copy()
    if subset_df.empty:
        print(f"Subsetted WVS data is empty for countries {countries}. Please check country codes and data.")
        return subset_df # Return empty df with correct columns if possible

    print(f"Subsetted WVS data for countries {countries}. Shape after country filter: {subset_df.shape}")

    for q_id in econ_q_ids:
        if q_id in subset_df.columns:
            subset_df[q_id] = pd.to_numeric(subset_df[q_id], errors='coerce')
        else:
            print(f"Warning: Economic question ID {q_id} not found in WVS subset.")


    if 'Q262' in subset_df.columns:
        subset_df['Age_Group'] = subset_df['Q262'].apply(map_age_to_group)
    else:
        subset_df['Age_Group'] = "Unknown"
        print("Warning: Q262 (Age) not found in selected columns, using 'Unknown' for Age_Group.")

    return subset_df

def generate_prompt_data_from_wvs_split(wvs_split_df, econ_questions_map_local, 
                                        training_prompt_fmt, inference_prompt_fmt,
                                        is_training_data=True):
    """
    Generates prompt data (either for training or for inference+ground_truth).
    If is_training_data is True, generates TARGET_DATA_FOR_BIAS.
    If False, generates INFERENCE_PROMPTS and GROUND_TRUTH_DISTRIBUTIONS.
    """
    output_prompts_or_examples = []
    ground_truth_distributions = {} # Only used if not is_training_data

    if wvs_split_df.empty:
        print(f"WVS split is empty. Cannot generate {'training' if is_training_data else 'evaluation'} data.")
        if not is_training_data:
            return [], {}
        else:
            return []
            
    print(f"Processing {len(wvs_split_df)} WVS rows for {'training' if is_training_data else 'evaluation'} data...")

    for index, row in wvs_split_df.iterrows():
        country = row['B_COUNTRY_ALPHA']
        age_group = row.get('Age_Group', "Unknown") # Ensure Age_Group exists

        for q_id, q_info in econ_questions_map_local.items():
            if q_id in row: # Check if the economic question column exists in the row/df
                response_value = row[q_id]
                if pd.notna(response_value) and 0 < response_value <= 10: # WVS valid responses are >0
                    response_int = int(response_value)
                    response_str = str(response_int)
                    question_key = q_info["key"]

                    if is_training_data:
                        training_example = training_prompt_fmt.format(country, age_group, question_key, response_str)
                        output_prompts_or_examples.append(training_example)
                    else:
                        inference_prompt = inference_prompt_fmt.format(country, age_group, question_key)
                        if inference_prompt not in output_prompts_or_examples:
                             output_prompts_or_examples.append(inference_prompt)

                        if inference_prompt not in ground_truth_distributions:
                            ground_truth_distributions[inference_prompt] = Counter()
                        ground_truth_distributions[inference_prompt][response_int] += 1
            # else:
            #     print(f"Warning: Question ID {q_id} not found for row {index}. Skipping.")


    if not is_training_data:
        # Normalize ground truth distributions
        for prompt in ground_truth_distributions:
            total_counts = sum(ground_truth_distributions[prompt].values())
            if total_counts > 0:
                for val in ground_truth_distributions[prompt]:
                    ground_truth_distributions[prompt][val] /= total_counts
        return output_prompts_or_examples, ground_truth_distributions
    else:
        return output_prompts_or_examples