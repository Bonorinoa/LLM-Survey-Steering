# scripts/run_experiment.py

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import re 

# Imports from our package
from llm_survey_steering.config import ProjectConfig
from llm_survey_steering.data_processing.wvs_processor import (
    load_wvs_data,
    preprocess_wvs_data,
    generate_prompt_data_from_wvs_split # Will be updated later to use tokenizer
)
from llm_survey_steering.models.architecture import load_models_and_tokenizer
from llm_survey_steering.training.trainer import prepare_dataset, train_reweighting_model
from llm_survey_steering.evaluation.evaluator import evaluate_survey_responses
from llm_survey_steering.evaluation.plots import plot_loss, plot_response_distributions
from sklearn.model_selection import train_test_split # Moved here for clarity

# from root LLM-Survey-Steering folder, run 'python scripts/run_experiment.py' on terminal to get results for WVS dataset.

def main():
    """
    Main function to run the LLM survey steering experiment.
    Orchestrates data loading, model initialization, training, and evaluation.
    """
    print("--- Initializing Configuration ---")
    config = ProjectConfig()
    # ---- Example: Set up for a chat model (uncomment and modify to test) ----
    # config.BASE_MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf" # Requires HuggingFace login & access
    # config.MODEL_TYPE = "chat" # Or "auto"
    # ---- End Example ----
    
    config._determine_model_type_if_needed() # Call to auto-set MODEL_TYPE if it's "auto"
    config.setup_seeds()
    print(f"Using device: {config.device}")
    print(f"Base model: {config.BASE_MODEL_NAME}, Model type: {config.MODEL_TYPE}")

    # --- 1. Initialize Models & Tokenizer (Tokenizer needed for prompt generation if chat model) ---
    print("\n--- Initializing Models & Tokenizer ---")
    # load_models_and_tokenizer now sets tokenizer and vocab_size on the config object
    base_model, reweighting_model = load_models_and_tokenizer(config)
    
    # Now config.tokenizer and config.vocab_size are available

    # --- 2. Data Processing ---
    print("\n--- Starting WVS Data Processing ---")
    wvs_data_full = load_wvs_data(config.WVS_FILE_PATH)

    if wvs_data_full is None:
        print("Failed to load WVS data. Cannot proceed.")
        return

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

    train_wvs_df, test_wvs_df = train_test_split(
        wvs_subset_preprocessed,
        test_size=config.TEST_SET_SIZE,
        random_state=config.RANDOM_STATE_SPLIT
    )
    print(f"WVS data split: {len(train_wvs_df)} training rows, {len(test_wvs_df)} test rows.")

    # Generate TARGET_DATA_FOR_BIAS
    # This function will be updated to accept and use config.tokenizer and config.MODEL_TYPE
    config.TARGET_DATA_FOR_BIAS = generate_prompt_data_from_wvs_split(
        wvs_split_df=train_wvs_df,
        config_obj=config, # Pass the whole config object
        # tokenizer=config.tokenizer, # Pass tokenizer explicitly
        # model_type=config.MODEL_TYPE, # Pass model_type explicitly
        is_training_data=True
    )
    print(f"Generated {len(config.TARGET_DATA_FOR_BIAS)} training examples for reweighting model.")
    if config.TARGET_DATA_FOR_BIAS:
        print("Sample training example:", config.TARGET_DATA_FOR_BIAS[0])
    else:
        print("No training examples generated. Check data and processing steps.")

    # Generate INFERENCE_PROMPTS and GROUND_TRUTH_DISTRIBUTIONS
    config.GENERATION_PROMPTS_SURVEY, config.GROUND_TRUTH_DISTRIBUTIONS = generate_prompt_data_from_wvs_split(
        wvs_split_df=test_wvs_df,
        config_obj=config, # Pass the whole config object
        # tokenizer=config.tokenizer,
        # model_type=config.MODEL_TYPE,
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


    # --- 3. Reweighting Model Training ---
    loss_history = []
    if not config.TARGET_DATA_FOR_BIAS:
        print("TARGET_DATA_FOR_BIAS is empty. Skipping reweighting model training.")
    else:
        print("\n--- Preparing Training Dataloader for Reweighting Model ---")
        train_dataloader = prepare_dataset(
            texts=config.TARGET_DATA_FOR_BIAS,
            tokenizer=config.tokenizer, # Use tokenizer from config
            max_length=config.MAX_SEQ_LENGTH,
            batch_size=config.BATCH_SIZE
        )
        
        if not train_dataloader.dataset or len(train_dataloader.dataset) == 0:
            print("Training dataloader is empty. Skipping reweighting model training.")
        else:
            optimizer = optim.AdamW(reweighting_model.parameters(), lr=config.LEARNING_RATE)
            criterion = nn.CrossEntropyLoss(ignore_index=config.tokenizer.pad_token_id if config.tokenizer.pad_token_id is not None else -100)

            loss_history = train_reweighting_model(
                reweighting_model=reweighting_model,
                base_model=base_model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                #vocab_size=config.vocab_size, # Use vocab_size from config
                config_obj=config,
                tokenizer=config.tokenizer # Pass tokenizer for labels padding ID
            )
            print("\n--- Analysis: Reweighting Model Training Loss ---")
            if loss_history:
                plot_loss(loss_history)

    # --- 4. Evaluation ---
    if not config.GENERATION_PROMPTS_SURVEY:
        print("\nNo inference prompts available. Skipping survey response evaluation.")
    else:
        evaluation_df = evaluate_survey_responses(
            config_obj=config, # Contains tokenizer and other params
            base_model=base_model,
            reweighting_model=reweighting_model if loss_history else None,
            ground_truth_distributions_map=config.GROUND_TRUTH_DISTRIBUTIONS
        )

        if not evaluation_df.empty:
            print("\n--- Quantitative Evaluation Metrics (Survey Responses) ---")
            
            def create_prompt_key(prompt_str):
                try:
                    # This parsing might need adjustment if prompt structure changes for chat models
                    if "Demographics: " in prompt_str and " SurveyContext: " in prompt_str:
                        demo_part = prompt_str.split("Demographics: ")[1].split(" SurveyContext:")[0]
                        context_part = prompt_str.split("SurveyContext: ")[1].split(". My view")[0]
                        return f"{demo_part} | {context_part}"
                    else: # Fallback for differently structured (e.g. chat template) prompts
                        return prompt_str[:70].replace('\n', ' ') + "..."
                except Exception:
                    return prompt_str[:70].replace('\n', ' ') + "..."


            display_df = evaluation_df.copy()
            display_df["prompt_key"] = display_df["prompt"].apply(create_prompt_key)
            
            cols_to_display = [
                "prompt_key", "base_mean_response", "plugin_mean_response", "ground_truth_mean_response",
                "js_base_vs_gt", "js_plugin_vs_gt",
                "num_valid_base", "num_valid_plugin"
            ]
            actual_cols_to_display = [col for col in cols_to_display if col in display_df.columns]
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_colwidth', 200)
            print(display_df[actual_cols_to_display])

            plot_response_distributions(evaluation_df)

            print("\n--- Qualitative Examples (Survey Responses, First 3 Prompts) ---")
            for i in range(min(3, len(evaluation_df))):
                row = evaluation_df.iloc[i]
                print(f"\nPrompt:\n{row['prompt']}") # Print full prompt for chat models
                print(f"  Base Model Response Distribution (Counts): "
                      f"{row.get('base_response_distribution_counts', 'N/A')}")
                print(f"  Plugin Model Response Distribution (Counts): "
                      f"{row.get('plugin_response_distribution_counts', 'N/A')}")
                print(f"  Ground Truth Distribution (Probs): {row.get('ground_truth_distribution_probs', 'N/A')}")

    print("\n--- Experiment End ---")

if __name__ == '__main__':
    main()