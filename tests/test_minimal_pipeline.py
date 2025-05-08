import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import random # For synthetic data generation
import sys
import os
from sklearn.model_selection import train_test_split # For splitting synthetic data

# Adjust Python path to include the project root if running from tests/ directory
# This allows imports from llm_survey_steering
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from your package
from llm_survey_steering.config import ProjectConfig
from llm_survey_steering.data_processing import (
    generate_prompt_data_from_wvs_split, # We'll use this with our synthetic data
    map_age_to_group # For processing synthetic age
)
from llm_survey_steering.models.architecture import load_models_and_tokenizer
from llm_survey_steering.training.trainer import prepare_dataset, train_reweighting_model
from llm_survey_steering.evaluation.evaluator import evaluate_survey_responses
# Plots are optional for a minimal test, can be commented out if they cause issues in CI/headless env
# from llm_survey_steering.evaluation.plots import plot_loss 


def generate_minimal_synthetic_df(num_rows=20, config_obj: ProjectConfig = None) -> pd.DataFrame:
    """
    Generates a very small, minimal DataFrame mimicking WVS structure for quick testing.
    Uses question IDs and countries from the provided or default config.
    """
    print(f"Generating {num_rows} rows of minimal synthetic data...")
    if config_obj is None: # Should not happen if called correctly
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
    df['Age_Group'] = df['Q262'].apply(map_age_to_group) # Process age into groups
    print("Minimal synthetic data generated.")
    return df

def run_pipeline_for_test_model(base_model_name: str, model_type: str):
    """
    Runs a minimal version of the pipeline for a given model configuration.
    """
    print(f"\n--- RUNNING MINIMAL TEST FOR: {base_model_name} (Type: {model_type}) ---")
    
    config = ProjectConfig()
    
    # Apply minimal test overrides
    config.minimal_test_mode = True # To ensure internal logic uses minimal settings if any
    config.EPOCHS = 1
    config.NUM_PREDICTIONS_PER_PROMPT_EVAL = 2 # Very few predictions
    config.BATCH_SIZE = 2 
    config.minimal_test_num_prompts_eval = 2 # Limit prompts in eval
    
    # Override model settings
    config.BASE_MODEL_NAME = base_model_name
    config.MODEL_TYPE = model_type # "causal" or "chat"

    config._determine_model_type_if_needed() # Important if MODEL_TYPE="auto" was set
    config.setup_seeds()
    print(f"Using device: {config.device}")

    print("Initializing models & tokenizer...")
    base_model, reweighting_model = load_models_and_tokenizer(config)

    # === ADD THIS SECTION FOR CHAT TEMPLATE (START) ===
    if config.MODEL_TYPE == "chat" and config.tokenizer.chat_template is None:
        print(f"WARNING: Tokenizer for {config.BASE_MODEL_NAME} does not have a default chat template. "
            "Applying a basic one for testing chat logic.")
        # A very basic template suitable for many models if they don't have one.
        # This example uses a simple User/Assistant structure.
        # For more complex roles or system prompts, this would need to be more elaborate.
        config.tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                    "User: {{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}"
                    "Assistant: {{ message['content'] }}{{ eos_token if not loop.last }}\n" # Add EOS for assistant turns
                "{% else %}"
                    "{{ message['role'] }}: {{ message['content'] }}\n"
                "{% endif %}"
            "{% endfor %}"
            # "{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}" # Handled by add_generation_prompt=True in apply_chat_template
            #     "Assistant:\n"
            # "{% endif %}"
        )
        print(f"Applied custom chat template: {config.tokenizer.chat_template}")
    # === ADD THIS SECTION FOR CHAT TEMPLATE (END) ===

    # 2. Generate & Process Synthetic Data for this run
    synthetic_df = generate_minimal_synthetic_df(num_rows=30, config_obj=config) # More rows for better split
    
    # Stratify if possible (though less critical for such small synthetic data)
    stratify_col = None
    if 'B_COUNTRY_ALPHA' in synthetic_df.columns and synthetic_df['B_COUNTRY_ALPHA'].nunique() > 1:
        if synthetic_df['B_COUNTRY_ALPHA'].value_counts().min() >= 2: # sklearn needs at least 2 per class
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
        print("CRITICAL: No training examples generated from synthetic data. Check synthetic data and prompt generation.")
        return False # Stop this test run

    unique_prompts_for_eval, gt_distributions_for_eval = generate_prompt_data_from_wvs_split(
        wvs_split_df=eval_df, config_obj=config, is_training_data=False
    )
    
    # Further trim eval prompts for minimal test mode
    if unique_prompts_for_eval:
        prompts_to_keep = sorted(list(set(unique_prompts_for_eval))) # Unique prompts
        prompts_to_keep = prompts_to_keep[:min(len(prompts_to_keep), config.minimal_test_num_prompts_eval)]
        config.GENERATION_PROMPTS_SURVEY = prompts_to_keep
        config.GROUND_TRUTH_DISTRIBUTIONS = {
            p: gt_distributions_for_eval[p] for p in prompts_to_keep if p in gt_distributions_for_eval
        }
    else:
        config.GENERATION_PROMPTS_SURVEY = []
        config.GROUND_TRUTH_DISTRIBUTIONS = {}

    print(f"Using {len(config.GENERATION_PROMPTS_SURVEY)} prompts for evaluation.")
    if not config.GENERATION_PROMPTS_SURVEY:
        print("WARNING: No evaluation prompts generated. Evaluation will be skipped.")


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
            # if loss_history:
            #     plot_loss(loss_history) # Optional plot
        else:
            print("Training dataloader empty, skipping training.")
    else:
        print("No training data, skipping training.")

    # 5. Evaluation
    if config.GENERATION_PROMPTS_SURVEY:
        print("Evaluating survey responses...")
        evaluation_df = evaluate_survey_responses(
            config_obj=config,  # Pass the full config object
            tokenizer=config.tokenizer, # Explicitly pass the tokenizer
            base_model=base_model,
            reweighting_model=reweighting_model if loss_history else None,
            ground_truth_distributions_map=config.GROUND_TRUTH_DISTRIBUTIONS # Correct
        )
        if not evaluation_df.empty:
            print("Minimal Evaluation Results (first row):")
            print(evaluation_df.head(1)[["prompt", "base_mean_response", "plugin_mean_response", "js_base_vs_gt", "js_plugin_vs_gt"]])
        else:
            print("Evaluation produced an empty DataFrame.")
    else:
        print("No prompts for evaluation, skipping.")
        
    print(f"--- MINIMAL TEST COMPLETED FOR: {base_model_name} ---")
    return True


if __name__ == "__main__":
    print("Starting minimal pipeline test suite...")
    
    test_passed_causal = False
    try:
        # Test 1: Small Causal LM (distilgpt2 is smaller and faster than gpt2-medium for testing)
        test_passed_causal = run_pipeline_for_test_model(base_model_name="distilgpt2", model_type="causal")
    except Exception as e:
        print(f"ERROR during causal model test (distilgpt2): {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50 + "\n")

    test_passed_chat = False
    try:
        # Test 2: Small (non-gated) Chat LM
        # HuggingFaceH4/tiny-llama-1.1b-chat-v1.0 is a good small, open chat model.
        # It requires internet to download on first run.
        # If internet is an issue or for a pure logic test of chat formatting,
        # you can use "distilgpt2" with model_type="chat".
        print("Attempting chat model test. This might download model files if not cached.")
        print("If using a gated model, ensure you have run 'huggingface-cli login' previously.")
        # test_passed_chat = run_pipeline_for_test_model(base_model_name="HuggingFaceH4/tiny-llama-1.1b-chat-v1.0", model_type="chat")
        
        # Fallback for simpler test: Use distilgpt2 to test chat formatting logic
        test_passed_chat = run_pipeline_for_test_model(base_model_name="distilgpt2", model_type="chat")
        
    except Exception as e:
        print(f"ERROR during chat model test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50 + "\nMinimal Test Suite Summary:")
    print(f"Causal Model Test (distilgpt2) Passed: {test_passed_causal}")
    print(f"Chat Model Test (distilgpt2 with chat logic) Passed: {test_passed_chat}")
    
    if test_passed_causal and test_passed_chat:
        print("All minimal tests seem to have run.")
    else:
        print("One or more minimal tests encountered issues.")