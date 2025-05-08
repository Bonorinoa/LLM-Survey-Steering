# scripts/run_experiment.py
import sys
import os
import torch # Try importing torch first
import numpy
import transformers # Then transformers

print("--- Environment Verification ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Path: {torch.__file__}")
print(f"NumPy Version: {numpy.__version__}")
print(f"NumPy Path: {numpy.__file__}")
print(f"Transformers Version: {transformers.__version__}")
print(f"Transformers Path: {transformers.__file__}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"Current Working Directory: {os.getcwd()}")
print("sys.path:")
for p in sys.path:
    print(f"  {p}")
print("--- End Environment Verification ---")

# Check for torch.compiler explicitly right after import
if hasattr(torch, 'compiler'):
    print("torch.compiler IS available.")
else:
    print("torch.compiler IS NOT available. This is the problem.")
    # You could even try a direct import if supported by your torch version structure
    try:
        import torch.compiler
        print("Successfully imported torch.compiler directly.")
    except ImportError as e:
        print(f"Failed to import torch.compiler directly: {e}")
    except AttributeError as e:
        print(f"AttributeError when trying to access or import torch.compiler: {e}")
        
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import re # For creating prompt_key in display_df

# Imports from our package
from llm_survey_steering.config import ProjectConfig, ECON_QUESTIONS_MAP # Also import global constants if needed by data funcs
from llm_survey_steering.data_processing.wvs_processor import (
    load_wvs_data,
    preprocess_wvs_data,
    generate_prompt_data_from_wvs_split
)
from llm_survey_steering.models.architecture import load_models_and_tokenizer
from llm_survey_steering.training.trainer import prepare_dataset, train_reweighting_model
from llm_survey_steering.evaluation.evaluator import evaluate_survey_responses
from llm_survey_steering.evaluation.plots import plot_loss, plot_response_distributions


def main():
    print("--- Initializing Configuration ---")
    config = ProjectConfig()
    config.setup_seeds() # Set random seeds
    print(f"Using device: {config.device}")
    
    # --- 1. Data Processing ---
    print("\n--- Starting WVS Data Processing ---")
    wvs_data_full = load_wvs_data(config.WVS_FILE_PATH)

    if wvs_data_full is None:
        print("Failed to load WVS data. Cannot proceed.")
        return # Changed from exit() to return for better script flow

    wvs_subset_preprocessed = preprocess_wvs_data(
        wvs_data_full,
        countries=config.TARGET_COUNTRIES,
        general_vars=config.GENERAL_VARIABLES,
        demo_vars=config.DEMOGRAPHIC_VARIABLES_IDS,
        econ_q_ids=config.ECONOMIC_QUESTION_IDS
    )

    if wvs_subset_preprocessed.empty:
        print("WVS subset is empty after preprocessing. Cannot proceed.")
        return

    # Split WVS data into train and test FOR THE ROWS
    # Ensure train_test_split is imported if not already via pandas or other modules
    from sklearn.model_selection import train_test_split
    train_wvs_df, test_wvs_df = train_test_split(
        wvs_subset_preprocessed,
        test_size=config.TEST_SET_SIZE,
        random_state=config.RANDOM_STATE_SPLIT
    )
    print(f"WVS data split: {len(train_wvs_df)} training rows, {len(test_wvs_df)} test rows.")

    # Generate TARGET_DATA_FOR_BIAS from the WVS training split
    # Pass necessary format strings and maps from config
    config.TARGET_DATA_FOR_BIAS = generate_prompt_data_from_wvs_split(
        train_wvs_df,
        econ_questions_map_local=config.ECON_QUESTIONS_MAP, # Pass from config
        training_prompt_fmt=config.TRAINING_PROMPT_FORMAT,
        inference_prompt_fmt=config.INFERENCE_PROMPT_FORMAT, # Not used here but good for consistency
        is_training_data=True
    )
    print(f"Generated {len(config.TARGET_DATA_FOR_BIAS)} training examples for reweighting model.")
    if config.TARGET_DATA_FOR_BIAS:
        print("Sample training example:", config.TARGET_DATA_FOR_BIAS[0])
    else:
        print("No training examples generated. Check data and processing steps.")


    # Generate INFERENCE_PROMPTS and GROUND_TRUTH_DISTRIBUTIONS from the WVS test split
    config.GENERATION_PROMPTS_SURVEY, config.GROUND_TRUTH_DISTRIBUTIONS = generate_prompt_data_from_wvs_split(
        test_wvs_df,
        econ_questions_map_local=config.ECON_QUESTIONS_MAP, # Pass from config
        training_prompt_fmt=config.TRAINING_PROMPT_FORMAT, # Not used here
        inference_prompt_fmt=config.INFERENCE_PROMPT_FORMAT,
        is_training_data=False
    )
    print(f"Generated {len(config.GENERATION_PROMPTS_SURVEY)} unique inference prompts for evaluation.")
    if config.GENERATION_PROMPTS_SURVEY:
        sample_inf_prompt = config.GENERATION_PROMPTS_SURVEY[0]
        print("Sample inference prompt:", sample_inf_prompt)
        if config.GROUND_TRUTH_DISTRIBUTIONS.get(sample_inf_prompt):
            print("Ground truth for sample prompt:", dict(config.GROUND_TRUTH_DISTRIBUTIONS[sample_inf_prompt]))
    else:
        print("No inference prompts generated. Check data and processing steps.")

    # --- 2. Model Initialization ---
    print("\n--- Initializing Models ---")
    tokenizer, base_model, reweighting_model, vocab_size = load_models_and_tokenizer(config)

    # --- 3. Reweighting Model Training ---
    loss_history = []
    if not config.TARGET_DATA_FOR_BIAS:
        print("TARGET_DATA_FOR_BIAS is empty. Skipping reweighting model training.")
    else:
        print("\n--- Preparing Training Dataloader for Reweighting Model ---")
        train_dataloader = prepare_dataset(
            texts=config.TARGET_DATA_FOR_BIAS,
            tokenizer=tokenizer,
            max_length=config.MAX_SEQ_LENGTH,
            batch_size=config.BATCH_SIZE
        )
        
        if not train_dataloader.dataset or len(train_dataloader.dataset) == 0:
            print("Training dataloader is empty. Skipping reweighting model training.")
        else:
            optimizer = optim.AdamW(reweighting_model.parameters(), lr=config.LEARNING_RATE)
            # Ignore padding tokens (-100) and other special tokens if necessary
            criterion = nn.CrossEntropyLoss(ignore_index=-100) 

            loss_history = train_reweighting_model(
                reweighting_model=reweighting_model,
                base_model=base_model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                vocab_size=vocab_size,
                config_obj=config,
                tokenizer=tokenizer # Pass tokenizer for label padding ID
            )
            print("\n--- Analysis: Reweighting Model Training Loss ---")
            if loss_history:
                plot_loss(loss_history) # Add save_path= if desired

    # --- 4. Evaluation ---
    if not config.GENERATION_PROMPTS_SURVEY:
        print("\nNo inference prompts available. Skipping survey response evaluation.")
    else:
        evaluation_df = evaluate_survey_responses(
            config_obj=config,
            tokenizer=tokenizer,
            base_model=base_model,
            reweighting_model=reweighting_model if loss_history else None, # Pass reweighting model only if trained
            ground_truth_distributions_map=config.GROUND_TRUTH_DISTRIBUTIONS
        )

        if not evaluation_df.empty:
            print("\n--- Quantitative Evaluation Metrics (Survey Responses) ---")
            
            # Create a more readable prompt key for display
            def create_prompt_key(prompt_str):
                try:
                    demo_part = prompt_str.split("Demographics: ")[1].split(" SurveyContext:")[0]
                    context_part = prompt_str.split("SurveyContext: ")[1].split(". My view")[0]
                    return f"{demo_part} | {context_part}"
                except:
                    return prompt_str[:70] # Fallback

            display_df = evaluation_df.copy()
            display_df["prompt_key"] = display_df["prompt"].apply(create_prompt_key)
            
            cols_to_display = [
                "prompt_key", "base_mean_response", "plugin_mean_response", "ground_truth_mean_response",
                "js_base_vs_gt", "js_plugin_vs_gt",
                "num_valid_base", "num_valid_plugin"
            ]
            # Ensure all columns in cols_to_display exist before trying to select them
            actual_cols_to_display = [col for col in cols_to_display if col in display_df.columns]
            
            # Configure pandas display options for better readability
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_colwidth', 200)

            print(display_df[actual_cols_to_display])

            plot_response_distributions(evaluation_df) # Add save_path_prefix= if desired

            print("\n--- Qualitative Examples (Survey Responses, First 3 Prompts) ---")
            for i in range(min(3, len(evaluation_df))):
                row = evaluation_df.iloc[i]
                print(f"\nPrompt: {row['prompt']}")
                print(f"  Base Model Response Distribution (Counts): "
                      f"{row.get('base_response_distribution_counts', 'N/A')}")
                print(f"  Plugin Model Response Distribution (Counts): "
                      f"{row.get('plugin_response_distribution_counts', 'N/A')}")
                print(f"  Ground Truth Distribution (Probs): {row.get('ground_truth_distribution_probs', 'N/A')}")
                print(f"  Sample Raw Base Responses: {row.get('raw_base_responses_sample', 'N/A')}")
                print(f"  Sample Raw Plugin Responses: {row.get('raw_plugin_responses_sample', 'N/A')}")

    print("\n--- Experiment End ---")

if __name__ == '__main__':
    main()
