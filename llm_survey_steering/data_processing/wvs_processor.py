# llm_survey_steering/data_processing/wvs_processor.py

import pandas as pd
import random
from sklearn.model_selection import train_test_split # Keep for direct use here if preferred
from collections import Counter
# No direct import of config constants like TRAINING_PROMPT_FORMAT if passed via config_obj

def map_age_to_group(age):
    """
    Simple function to categorize age into groups.

    Args:
        age (float or int): The age of the respondent.

    Returns:
        str: Age group ("Young", "Mid", "Old", or "Unknown").
    """
    if pd.isna(age):
        return "Unknown"
    try:
        age = int(age)
        if age < 30:
            return "Young"
        elif age < 50:
            return "Mid"
        else:
            return "Old"
    except ValueError:
        return "Unknown"


def load_wvs_data(file_path: str) -> pd.DataFrame | None:
    """
    Loads WVS data from the specified CSV file path.

    Args:
        file_path (str): Path to the WVS CSV file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if file not found.
    """
    try:
        wvs_full = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded WVS data. Shape: {wvs_full.shape}")
        return wvs_full
    except FileNotFoundError:
        print(f"Error: WVS data file not found at {file_path}. Please check the path.")
        return None


def preprocess_wvs_data(df: pd.DataFrame, countries: list, general_vars: list,
                        demo_vars: list, econ_q_ids: list) -> pd.DataFrame:
    """
    Subsets and preprocesses the WVS dataframe.

    Args:
        df (pd.DataFrame): The full WVS DataFrame.
        countries (list): List of target country codes (e.g., 'USA').
        general_vars (list): List of general variable IDs.
        demo_vars (list): List of demographic variable IDs (e.g., 'Q262' for age).
        econ_q_ids (list): List of economic question IDs.

    Returns:
        pd.DataFrame: The preprocessed subset DataFrame.
    """
    columns_to_select = general_vars + demo_vars + econ_q_ids
    columns_to_select = [col for col in columns_to_select if col in df.columns]

    if not df['B_COUNTRY_ALPHA'].isin(countries).any():
        print(f"Warning: None of the target countries {countries} found in B_COUNTRY_ALPHA. Returning empty DataFrame.")
        return pd.DataFrame(columns=columns_to_select + ['Age_Group'])

    subset_df = df[df['B_COUNTRY_ALPHA'].isin(countries)][columns_to_select].copy()
    if subset_df.empty:
        print(f"Subsetted WVS data is empty for countries {countries}.")
        return subset_df

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
        print("Warning: Q262 (Age) not found, using 'Unknown' for Age_Group.")
    return subset_df


def _format_prompt_causal(config_obj, country, age_group, question_key, response_str=None, is_training=True):
    """Helper to format prompts for causal LMs."""
    if is_training:
        return config_obj.CAUSAL_LM_TRAINING_PROMPT_FORMAT.format(country, age_group, question_key, response_str)
    else:
        return config_obj.CAUSAL_LM_INFERENCE_PROMPT_FORMAT.format(country, age_group, question_key)

def _format_prompt_chat(tokenizer, country, age_group, question_key, response_str=None, is_training=True):
    """
    Helper to format prompts for chat LMs using tokenizer.apply_chat_template.
    """
    user_content = f"Demographics: Country_{country}_Age_{age_group}. SurveyContext: {question_key}. My view on the scale is:"
    
    if is_training:
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response_str if response_str is not None else ""}
        ]
        # For training, we want the full exchange tokenized.
        # add_generation_prompt=False ensures it doesn't add a trailing assistant prompt if not desired for training data.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else: # For inference
        messages = [
            {"role": "user", "content": user_content}
        ]
        # add_generation_prompt=True adds the model-specific tokens to signal it should generate (e.g., "<|assistant|>")
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_prompt_data_from_wvs_split(wvs_split_df: pd.DataFrame,
                                        config_obj, # Contains tokenizer, MODEL_TYPE, ECON_QUESTIONS_MAP, prompt formats
                                        is_training_data: bool = True):
    """
    Generates prompt data for training or inference, adapting to model type.

    Args:
        wvs_split_df (pd.DataFrame): The WVS data split (train or test).
        config_obj (ProjectConfig): Configuration object containing tokenizer, model_type, etc.
        is_training_data (bool): True to generate training examples, False for inference prompts
                                 and ground truth distributions.

    Returns:
        If is_training_data is True: list[str] (list of training examples)
        If is_training_data is False: tuple (list[str], dict) 
                                         (list of inference prompts, ground_truth_distributions)
    """
    output_prompts_or_examples = []
    ground_truth_distributions = {} # Only used if not is_training_data

    if wvs_split_df.empty:
        print(f"WVS split is empty. Cannot generate {'training' if is_training_data else 'evaluation'} data.")
        return ([], {}) if not is_training_data else []
            
    print(f"Processing {len(wvs_split_df)} WVS rows for {'training' if is_training_data else 'evaluation'} data, model_type: {config_obj.MODEL_TYPE}.")

    for index, row in wvs_split_df.iterrows():
        country = row['B_COUNTRY_ALPHA']
        age_group = row.get('Age_Group', "Unknown")

        for q_id, q_info in config_obj.ECON_QUESTIONS_MAP.items():
            if q_id in row:
                response_value = row[q_id]
                if pd.notna(response_value) and 0 < response_value <= 10:
                    response_int = int(response_value)
                    response_str = str(response_int) # For causal LM format and chat assistant response
                    question_key = q_info["key"]

                    prompt_text = ""
                    if config_obj.MODEL_TYPE == "causal":
                        prompt_text = _format_prompt_causal(config_obj, country, age_group, question_key, 
                                                            response_str if is_training_data else None, 
                                                            is_training=is_training_data)
                    elif config_obj.MODEL_TYPE == "chat":
                        if config_obj.tokenizer is None:
                            raise ValueError("Tokenizer must be available in config_obj for chat model prompt formatting.")
                        prompt_text = _format_prompt_chat(config_obj.tokenizer, country, age_group, question_key,
                                                          response_str if is_training_data else None,
                                                          is_training=is_training_data)
                    else:
                        raise ValueError(f"Unsupported MODEL_TYPE: {config_obj.MODEL_TYPE}")

                    if is_training_data:
                        output_prompts_or_examples.append(prompt_text)
                    else: # For evaluation (inference prompts and ground truth)
                        # Store unique inference prompts
                        if prompt_text not in output_prompts_or_examples:
                             output_prompts_or_examples.append(prompt_text)
                        
                        # For ground truth, the key should be the inference prompt.
                        # If MODEL_TYPE is "chat", the prompt_text already has add_generation_prompt=True
                        inference_prompt_key = prompt_text 

                        if inference_prompt_key not in ground_truth_distributions:
                            ground_truth_distributions[inference_prompt_key] = Counter()
                        ground_truth_distributions[inference_prompt_key][response_int] += 1
            # else:
            #     print(f"Warning: Question ID {q_id} not found for row {index}. Skipping.")

    if not is_training_data:
        for prompt_key_for_gt in ground_truth_distributions:
            total_counts = sum(ground_truth_distributions[prompt_key_for_gt].values())
            if total_counts > 0:
                for val in ground_truth_distributions[prompt_key_for_gt]:
                    ground_truth_distributions[prompt_key_for_gt][val] /= total_counts
        return output_prompts_or_examples, ground_truth_distributions
    else:
        return output_prompts_or_examples
    

def generate_synthetic_wvs_data(num_rows: int, target_countries: list, 
                                econ_q_ids: list, demo_vars: list, 
                                general_vars: list) -> pd.DataFrame:
    """
    Generates a small synthetic DataFrame mimicking WVS structure for quick testing.
    """
    data = []
    if not target_countries: # Ensure there's at least one country to pick from
        target_countries = ['SYN'] 

    for _ in range(num_rows):
        row = {}
        # General vars
        if 'B_COUNTRY_ALPHA' in general_vars:
            row['B_COUNTRY_ALPHA'] = random.choice(target_countries)
        
        # Demo vars
        if 'Q262' in demo_vars: # Assuming Q262 is for Age
            row['Q262'] = random.randint(18, 75) 
        
        # Economic questions
        for q_id in econ_q_ids:
            row[q_id] = random.randint(1, 10)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Apply minimal preprocessing similar to preprocess_wvs_data
    # This part was simplified in the test script's local synthetic generator,
    # but it's good practice to have it here if this function were to be more general.
    # For the test script, it calls map_age_to_group separately.
    # Let's make this consistent with how the test script expects to use it,
    # or make it fully self-contained with age group mapping.

    # For direct use by the test script, which calls map_age_to_group afterwards,
    # we might not need to map age groups here.
    # However, if you intend this function to be a general utility, add it:
    # if 'Q262' in df.columns and 'Q262' in demo_vars: # If Q262 represents age
    #    df['Age_Group'] = df['Q262'].apply(map_age_to_group)
    # elif 'Age_Group' not in df.columns: # Ensure Age_Group column exists
    #    df['Age_Group'] = "Unknown"
        
    # Ensure numeric types for question IDs, similar to preprocess_wvs_data
    for q_id in econ_q_ids:
        if q_id in df.columns:
            df[q_id] = pd.to_numeric(df[q_id], errors='coerce')
            
    print(f"Generated {len(df)} rows of synthetic WVS-like data for wvs_processor.")
    return df
