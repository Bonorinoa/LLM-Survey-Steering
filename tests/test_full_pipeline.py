import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import random
import sys
import os
from sklearn.model_selection import train_test_split

# Adjust Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm_survey_steering.config import ProjectConfig
from llm_survey_steering.data_processing import (
    generate_prompt_data_from_wvs_split,
    map_age_to_group
)
from llm_survey_steering.models.architecture import load_models_and_tokenizer
from llm_survey_steering.training.trainer import prepare_dataset, train_reweighting_model
from llm_survey_steering.evaluation.evaluator import evaluate_survey_responses
# from llm_survey_steering.evaluation.plots import plot_loss # Optional

def generate_small_synthetic_df_for_real_test(num_rows=50, config_obj: ProjectConfig = None) -> pd.DataFrame:
    """Generates a small synthetic DataFrame for testing with real models."""
    print(f"Generating {num_rows} rows of synthetic data for real model test...")
    if config_obj is None:
        config_obj = ProjectConfig()

    data = []
    target_countries = config_obj.TARGET_COUNTRIES if config_obj.TARGET_COUNTRIES else ['SYN']
    econ_q_ids = config_obj.ECONOMIC_QUESTION_IDS
    
    for _ in range(num_rows):
        row = {
            'B_COUNTRY_ALPHA': random.choice(target_countries),
            'Q262': random.randint(18, 75), # Age
        }
        for q_id in econ_q_ids:
            row[q_id] = random.randint(1, 10)
        data.append(row)
    
    df = pd.DataFrame(data)
    df['Age_Group'] = df['Q262'].apply(map_age_to_group)
    print("Synthetic data for real model test generated.")
    return df

def run_real_model_test_pipeline(base_model_name: str, model_type: str):
    """
    Runs a slightly more demanding test pipeline with specified real models.
    Uses more synthetic data and more evaluation predictions than the minimal test.
    """
    print(f"\n--- RUNNING REAL MODEL TEST FOR: {base_model_name} (Type: {model_type}) ---")
    
    config = ProjectConfig() # Fresh config for each run
    
    # Test-specific overrides (still faster than a full run)
    config.EPOCHS = 1
    config.NUM_PREDICTIONS_PER_PROMPT_EVAL = 5 # More preds than minimal
    config.BATCH_SIZE = 2 # Keep batch size small
    config.minimal_test_num_prompts_eval = 3 # Evaluate a few more prompts
    
    # Set model details
    config.BASE_MODEL_NAME = base_model_name
    config.MODEL_TYPE = model_type

    config._determine_model_type_if_needed()
    config.setup_seeds()
    print(f"Using device: {config.device}")
    print(f"NOTE: This test might download model files for {base_model_name} if not cached, which can take time.")


    # 1. Initialize Models & Tokenizer
    print("Initializing models & tokenizer...")
    base_model, reweighting_model = load_models_and_tokenizer(config)
    # config.tokenizer and config.vocab_size are now set

    # If MODEL_TYPE is "chat", we now expect the real chat model's tokenizer to have a chat_template.
    # No manual template injection needed here unlike in the other minimal test for distilgpt2.
    if config.MODEL_TYPE == "chat" and config.tokenizer.chat_template is None:
        print(f"CRITICAL WARNING: Real chat model '{config.BASE_MODEL_NAME}' loaded, "
              "but its tokenizer.chat_template is None. This is unexpected for a dedicated chat model.")
        # This might indicate an issue with the model itself or its tokenizer_config.json
        # The pipeline might fail at prompt generation.

    # 2. Generate & Process Synthetic Data
    synthetic_df = generate_small_synthetic_df_for_real_test(num_rows=60, config_obj=config)
    
    stratify_col = None
    if 'B_COUNTRY_ALPHA' in synthetic_df.columns and synthetic_df['B_COUNTRY_ALPHA'].nunique() > 1:
        if synthetic_df['B_COUNTRY_ALPHA'].value_counts().min() >= 2:
             stratify_col = synthetic_df['B_COUNTRY_ALPHA']

    train_df, eval_df = train_test_split(
        synthetic_df, test_size=0.5, random_state=config.RANDOM_STATE_SPLIT, stratify=stratify_col
    )
    print(f"Synthetic data split: {len(train_df)} train, {len(eval_df)} eval rows.")

    # 3. Generate Prompts
    print("Generating prompts...")
    config.TARGET_DATA_FOR_BIAS = generate_prompt_data_from_wvs_split(
        wvs_split_df=train_df, config_obj=config, is_training_data=True
    )
    print(f"Generated {len(config.TARGET_DATA_FOR_BIAS)} training examples.")
    if not config.TARGET_DATA_FOR_BIAS:
        print("CRITICAL: No training examples generated. Halting test.")
        return False

    unique_prompts_for_eval, gt_distributions_for_eval = generate_prompt_data_from_wvs_split(
        wvs_split_df=eval_df, config_obj=config, is_training_data=False
    )
    
    if unique_prompts_for_eval:
        prompts_to_keep = sorted(list(set(unique_prompts_for_eval)))
        prompts_to_keep = prompts_to_keep[:min(len(prompts_to_keep), config.minimal_test_num_prompts_eval)]
        config.GENERATION_PROMPTS_SURVEY = prompts_to_keep
        config.GROUND_TRUTH_DISTRIBUTIONS = {
            p: gt_distributions_for_eval[p] for p in prompts_to_keep if p in gt_distributions_for_eval
        }
    else:
        config.GENERATION_PROMPTS_SURVEY = []
        config.GROUND_TRUTH_DISTRIBUTIONS = {}
    print(f"Using {len(config.GENERATION_PROMPTS_SURVEY)} prompts for evaluation.")

    # 4. Reweighting Model Training
    loss_history = []
    if config.TARGET_DATA_FOR_BIAS:
        print("Preparing training dataloader...")
        train_dataloader = prepare_dataset(
            texts=config.TARGET_DATA_FOR_BIAS, tokenizer=config.tokenizer,
            max_length=config.MAX_SEQ_LENGTH, batch_size=config.BATCH_SIZE
        )
        if train_dataloader.dataset and len(train_dataloader.dataset) > 0:
            optimizer = optim.AdamW(reweighting_model.parameters(), lr=config.LEARNING_RATE)
            criterion = nn.CrossEntropyLoss(
                ignore_index=config.tokenizer.pad_token_id if config.tokenizer.pad_token_id is not None else -100
            )
            print("Training reweighting model (1 epoch)...")
            loss_history = train_reweighting_model(
                reweighting_model, base_model, train_dataloader, optimizer, criterion,
                config, config.tokenizer
            )
        else:
            print("Training dataloader empty, skipping training.")
    else:
        print("No training data, skipping training.")

    # 5. Evaluation
    if config.GENERATION_PROMPTS_SURVEY:
        print("Evaluating survey responses...")
        evaluation_df = evaluate_survey_responses(
            config_obj=config,
            tokenizer=config.tokenizer,
            base_model=base_model,
            reweighting_model=reweighting_model if loss_history else None,
            ground_truth_distributions_map=config.GROUND_TRUTH_DISTRIBUTIONS
        )
        if not evaluation_df.empty:
            print("Evaluation Results (first row if available):")
            display_cols = ["prompt", "base_mean_response", "plugin_mean_response", "js_base_vs_gt", "js_plugin_vs_gt", "num_valid_base", "num_valid_plugin"]
            actual_display_cols = [col for col in display_cols if col in evaluation_df.columns]
            print(evaluation_df.head(1)[actual_display_cols])
            
            # Check if any valid responses were parsed, especially for the plugin model
            if evaluation_df["num_valid_plugin"].sum() == 0 and (reweighting_model is not None):
                print("WARNING: No valid responses were parsed for the plugin model during evaluation with this real model.")
            elif evaluation_df["num_valid_base"].sum() == 0:
                 print("WARNING: No valid responses were parsed for the base model during evaluation.")
        else:
            print("Evaluation produced an empty DataFrame.")
    else:
        print("No prompts for evaluation, skipping.")
        
    print(f"--- REAL MODEL TEST COMPLETED FOR: {base_model_name} ---")
    return True

if __name__ == "__main__":
    print("Starting real models pipeline test suite...")
    
    # Test 1: gpt2-medium (Causal)
    passed_gpt2_medium = False
    try:
        passed_gpt2_medium = run_real_model_test_pipeline(base_model_name="gpt2-medium", model_type="causal")
    except Exception as e:
        print(f"ERROR during gpt2-medium test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70 + "\n")

    # Test 2: vikp/instruct_llama_7b (Real Chat Model, non-gated)
    passed_tiny_llama = False
    try:
        passed_tiny_llama = run_real_model_test_pipeline(
            base_model_name="vikp/instruct_llama_7b", 
            model_type="chat"
        )
    except Exception as e:
        print(f"ERROR during TinyLlama-chat test: {e}")
        import traceback
        traceback.print_exc()
        if "authentication" in str(e).lower() or "gated" in str(e).lower():
            print("This error might be related to Hugging Face authentication or model access.")
            print("Ensure you can access this model, or try 'huggingface-cli login'.")

    print("\n" + "="*70 + "\nReal Models Test Suite Summary:")
    print(f"gpt2-medium Test Passed: {passed_gpt2_medium}")
    print(f"TinyLlama-chat Test Passed: {passed_tiny_llama}")
    
    if passed_gpt2_medium and passed_tiny_llama:
        print("All real model tests seem to have run successfully.")
    else:
        print("One or more real model tests encountered issues or did not complete.")