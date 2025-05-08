# llm_survey_steering/models/architecture.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

def load_models_and_tokenizer(config_obj):
    """Loads base model, reweighting model, and tokenizer based on config_obj."""
    print(f"Loading tokenizer for {config_obj.BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config_obj.BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    print(f"Loading base model: {config_obj.BASE_MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(config_obj.BASE_MODEL_NAME)
    base_model.to(config_obj.device)
    base_model.eval() # Set to evaluation mode
    for param in base_model.parameters():
        param.requires_grad = False # Freeze base model parameters
    print("Base model loaded and frozen.")

    print("Initializing reweighting model...")
    # Use base model's config for embedding dimensions etc.
    base_config_for_reweight = GPT2Config.from_pretrained(config_obj.BASE_MODEL_NAME)
    
    reweighting_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=config_obj.MAX_SEQ_LENGTH,
        n_embd=base_config_for_reweight.n_embd, # Match embedding dimension of base model
        n_layer=config_obj.REWEIGHTING_MODEL_N_LAYER,
        n_head=config_obj.REWEIGHTING_MODEL_N_HEAD,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    reweighting_model = GPT2LMHeadModel(reweighting_config)
    reweighting_model.to(config_obj.device)
    print(f"Reweighting model initialized with {config_obj.REWEIGHTING_MODEL_N_LAYER} layers and {config_obj.REWEIGHTING_MODEL_N_HEAD} heads.")
    
    return tokenizer, base_model, reweighting_model, vocab_size
