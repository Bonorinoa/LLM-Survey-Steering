# llm_survey_steering/generation/generator.py

import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper # For more controlled generation if needed

# config_obj, tokenizer, base_model, reweighting_model are passed as arguments

@torch.no_grad() # Ensure no gradients are computed during generation
def generate_survey_response(prompt_text, base_model, tokenizer, 
                             config_obj, # Pass the whole config object
                             reweighting_model=None):
    """
    Generates a survey response given a prompt, using either the base model
    or a combination of base and reweighting models.
    """
    base_model.eval() # Ensure base model is in eval mode
    if reweighting_model:
        reweighting_model.eval() # Ensure reweighting model is in eval mode

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt', truncation=True, max_length=config_obj.MAX_SEQ_LENGTH - config_obj.GENERATION_MAX_LEN_SURVEY).to(config_obj.device)
    
    # Ensure input_ids are not too long, leaving space for generation
    # This was slightly different in the original, let's ensure it's robust
    # max_input_len = config_obj.MAX_SEQ_LENGTH - config_obj.GENERATION_MAX_LEN_SURVEY
    # if input_ids.shape[1] > max_input_len:
    #     input_ids = input_ids[:, :max_input_len]
    
    generated_sequence_ids = input_ids.clone()

    for _ in range(config_obj.GENERATION_MAX_LEN_SURVEY):
        current_input_for_model = generated_sequence_ids
        
        # Prevent exceeding max_seq_length for models
        if current_input_for_model.shape[1] >= config_obj.MAX_SEQ_LENGTH:
            print(f"Warning: Input length {current_input_for_model.shape[1]} reached MAX_SEQ_LENGTH {config_obj.MAX_SEQ_LENGTH}. Stopping generation.")
            break

        # Get base model logits for the next token
        outputs_base = base_model(input_ids=current_input_for_model)
        # logits are for the *next* token, so shape is (batch_size, current_seq_len, vocab_size)
        # We need the logits for the very last token in the current sequence
        next_token_logits_base = outputs_base.logits[:, -1, :]
        
        # Apply temperature to base model logits
        probs_base_next = torch.softmax(next_token_logits_base / config_obj.GENERATION_TEMPERATURE_BASE, dim=-1)

        final_probs_next_token = probs_base_next

        if reweighting_model and config_obj.REWEIGHTING_STRENGTH_ALPHA > 0:
            outputs_reweighting = reweighting_model(input_ids=current_input_for_model)
            next_token_logits_reweighting = outputs_reweighting.logits[:, -1, :]
            
            # Apply temperature to reweighting model logits
            probs_reweighting_raw = torch.softmax(next_token_logits_reweighting / config_obj.GENERATION_TEMPERATURE_REWEIGHT, dim=-1)
            
            # Apply reweighting strength alpha (paper uses r_i^alpha, but here applied to probabilities for simplicity,
            # original script did probs_reweighting_raw ** reweighting_strength_alpha)
            # Let's stick to the original script's implementation for now:
            probs_reweighting_adjusted = probs_reweighting_raw ** config_obj.REWEIGHTING_STRENGTH_ALPHA
            
            # Combine probabilities: p_final = (p_base * p_reweight_adjusted) / norm
            combined_probs_product = probs_base_next * probs_reweighting_adjusted
            
            epsilon = 1e-9 # Small constant for numerical stability
            norm_factor = torch.sum(combined_probs_product, dim=-1, keepdim=True) + epsilon
            final_probs_next_token = combined_probs_product / norm_factor
        
        # Sample the next token ID from the final probability distribution
        try:
            next_token_id = torch.multinomial(final_probs_next_token, num_samples=1)
        except RuntimeError as e: # multinomial input can be all zeros if probabilities are tiny
            print(f"Warning: torch.multinomial failed ({e}). Using EOS token as fallback.")
            next_token_id = torch.tensor([[tokenizer.eos_token_id]], device=config_obj.device)
            
        # Append the chosen token to the sequence
        generated_sequence_ids = torch.cat((generated_sequence_ids, next_token_id), dim=1)
        
        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
    # Decode the generated part of the sequence (excluding the prompt)
    # input_ids.shape[1] gives the length of the original prompt
    generated_text = tokenizer.decode(generated_sequence_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text.strip()