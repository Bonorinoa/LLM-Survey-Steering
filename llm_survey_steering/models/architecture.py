# llm_survey_steering/models/architecture.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

def load_models_and_tokenizer(config_obj):
    """
    Loads base model, reweighting model, and tokenizer based on ProjectConfig.
    Sets tokenizer and vocab_size attributes on the config_obj.

    Args:
        config_obj (ProjectConfig): The project configuration object.

    Returns:
        tuple: (base_model, reweighting_model)
               The tokenizer and vocab_size are set on config_obj directly.
    """
    print(f"Loading tokenizer for {config_obj.BASE_MODEL_NAME}...")
    # For some chat models, like Llama, you might need to request access on Hugging Face Hub
    # and use `huggingface-cli login` before running the script.
    # trust_remote_code=True might be needed for some very new models.
    try:
        tokenizer = AutoTokenizer.from_pretrained(config_obj.BASE_MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer with trust_remote_code=True, trying without: {e}")
        tokenizer = AutoTokenizer.from_pretrained(config_obj.BASE_MODEL_NAME)


    if tokenizer.pad_token is None:
        # Common practice for models without a specific pad token (e.g., GPT-2, some chat models)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer pad_token was None, set to eos_token ({tokenizer.eos_token})")
    
    # For Llama 3 and some other models, setting padding_side = "left" for training causal LMs
    # can be beneficial, but for inference/generation, it's often handled by the generation kwargs.
    # For consistency in the reweighting model training, this might be relevant if inputs vary greatly.
    # tokenizer.padding_side = "left" # Common for causal LM fine-tuning.

    config_obj.tokenizer = tokenizer
    config_obj.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer loaded. Vocab size: {config_obj.vocab_size}. Pad token ID: {tokenizer.pad_token_id}")


    print(f"Loading base model: {config_obj.BASE_MODEL_NAME}...")
    # For large models, consider device_map="auto" or quantization if memory is an issue
    # (e.g., load_in_8bit=True with bitsandbytes library)
    base_model = AutoModelForCausalLM.from_pretrained(
        config_obj.BASE_MODEL_NAME,
        trust_remote_code=True # For models like Mixtral that might have custom code
        # torch_dtype=torch.bfloat16, # For faster inference on compatible GPUs, reduces memory
        # device_map="auto" # For multi-GPU or to offload parts to CPU
    )
    base_model.to(config_obj.device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    print(f"Base model {config_obj.BASE_MODEL_NAME} loaded and frozen on {config_obj.device}.")

    print("Initializing reweighting model...")
    try:
        # Attempt to get config from the loaded base model to match dimensions
        base_hf_config = base_model.config
        n_embd_reweight = base_hf_config.hidden_size # common attribute name
    except AttributeError:
        print("Warning: Could not get hidden_size from base_model.config. Falling back to GPT2Config for reweighting n_embd.")
        # Fallback: load config of base model name again to get its n_embd if base_model.config is not standard
        temp_base_config_for_reweight = GPT2Config.from_pretrained(config_obj.BASE_MODEL_NAME)
        n_embd_reweight = temp_base_config_for_reweight.n_embd

    reweighting_config = GPT2Config(
        vocab_size=config_obj.vocab_size, # Use vocab_size from the actual tokenizer
        n_positions=config_obj.MAX_SEQ_LENGTH,
        n_embd=n_embd_reweight, # Match embedding dimension
        n_layer=config_obj.REWEIGHTING_MODEL_N_LAYER,
        n_head=config_obj.REWEIGHTING_MODEL_N_HEAD,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id # Explicitly set pad_token_id for reweighting model
    )
    reweighting_model = GPT2LMHeadModel(reweighting_config)
    reweighting_model.to(config_obj.device)
    print(f"Reweighting model initialized with {config_obj.REWEIGHTING_MODEL_N_LAYER} layers, {config_obj.REWEIGHTING_MODEL_N_HEAD} heads on {config_obj.device}.")
    
    return base_model, reweighting_model