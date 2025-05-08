
# llm_survey_steering/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm # Use standard tqdm for scripts

# Note: tokenizer is passed as an argument, no direct import from models needed here if so.
# config_obj is also passed as an argument.

def prepare_dataset(texts, tokenizer, max_length, batch_size):
    """Prepares a TensorDataset and DataLoader from a list of texts."""
    all_input_ids, all_attention_masks = [], []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        all_input_ids.append(encoded['input_ids'])
        all_attention_masks.append(encoded['attention_mask'])
    
    if not all_input_ids: # Handle empty texts list
        print("Warning: No texts provided to prepare_dataset. Returning empty DataLoader.")
        # Create an empty TensorDataset and DataLoader
        # This requires defining what shape an empty tensor would have, or handling it upstream
        # For now, let's assume texts will not be empty if training is attempted.
        # Or, more robustly:
        empty_tensor_ids = torch.empty((0, max_length), dtype=torch.long)
        empty_tensor_mask = torch.empty((0, max_length), dtype=torch.long)
        dataset = TensorDataset(empty_tensor_ids, empty_tensor_mask)
        return DataLoader(dataset, batch_size=batch_size)


    input_ids = torch.cat(all_input_ids, dim=0)
    attention_masks = torch.cat(all_attention_masks, dim=0)
    
    dataset = TensorDataset(input_ids, attention_masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_reweighting_model(reweighting_model, base_model, dataloader, 
                            optimizer, criterion, # vocab_size no longer needed here if criterion is configured
                            config_obj, tokenizer): # Pass config_obj
    """Trains the reweighting model."""
    print("Starting training of reweighting model...")
    reweighting_model.train() 
    total_loss_history = []

    if not dataloader.dataset or len(dataloader.dataset) == 0:
        print("Warning: Training dataset is empty. Skipping training.")
        return total_loss_history

    # Determine the ignore_index value from the criterion itself, or from tokenizer if criterion doesn't expose it
    # However, criterion.ignore_index IS the correct way.
    # If criterion is not yet initialized with ignore_index (it is in run_experiment.py),
    # then tokenizer.pad_token_id is the fallback.
    # For safety, let's assume criterion HAS ignore_index set.
    current_ignore_index = criterion.ignore_index # This should be 50256 in your case, or -100 if pad_token_id was None

    for epoch in range(config_obj.EPOCHS):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config_obj.EPOCHS}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids, attention_mask = [b.to(config_obj.device) for b in batch]
            labels = input_ids.clone()
            
            # CRITICAL FIX: Use the same ignore_index that the criterion is using.
            # tokenizer.pad_token_id should be consistent with criterion.ignore_index
            # as set in run_experiment.py
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = current_ignore_index # Use current_ignore_index
            else:
                # This case should ideally not happen if pad_token is always set.
                # If pad_token_id was None, and ignore_index was set to -100 in criterion,
                # we'd need a different way to identify padding if it wasn't through a token ID.
                # But since we set pad_token = eos_token, pad_token_id is NOT None.
                pass # No explicit padding tokens to ignore by ID if pad_token_id is None

            optimizer.zero_grad()
            
            with torch.no_grad():
                outputs_base = base_model(input_ids=input_ids, attention_mask=attention_mask)
                probs_base = torch.softmax(outputs_base.logits, dim=-1) 
            
            outputs_reweighting = reweighting_model(input_ids=input_ids, attention_mask=attention_mask)
            probs_reweighting = torch.softmax(outputs_reweighting.logits, dim=-1)
            
            combined_probs_product = probs_base * probs_reweighting
            epsilon = 1e-9
            norm_factor = torch.sum(combined_probs_product, dim=-1, keepdim=True) + epsilon
            combined_probs_normalized = combined_probs_product / norm_factor
            final_logits_for_loss = torch.log(combined_probs_normalized + epsilon)

            # Pass config_obj.vocab_size to the criterion call's view
            loss = criterion(
                final_logits_for_loss[:, :-1, :].contiguous().view(-1, config_obj.vocab_size), # Use config_obj.vocab_size
                labels[:, 1:].contiguous().view(-1)
            )
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            else:
                print(f"Warning: NaN/Inf loss encountered at epoch {epoch+1}, batch {batch_idx}. Skipping batch update.")

            progress_bar.set_postfix({'loss': loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 'NaN/Inf'})
        
        avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        total_loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{config_obj.EPOCHS} - Average Loss: {avg_epoch_loss:.4f}")
        progress_bar.close()

    print("Training complete.")
    return total_loss_history