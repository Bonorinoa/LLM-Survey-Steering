
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
                            optimizer, criterion, vocab_size, 
                            config_obj, tokenizer):
    """Trains the reweighting model."""
    print("Starting training of reweighting model...")
    reweighting_model.train() # Set reweighting model to training mode
    total_loss_history = []

    if not dataloader.dataset or len(dataloader.dataset) == 0:
        print("Warning: Training dataset is empty. Skipping training.")
        return total_loss_history

    for epoch in range(config_obj.EPOCHS):
        epoch_loss = 0
        # Use standard tqdm for scripts
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config_obj.EPOCHS}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids, attention_mask = [b.to(config_obj.device) for b in batch]
            
            # Create labels: shift input_ids to the right for next token prediction
            # For Causal LM, labels are typically the input_ids themselves,
            # loss is calculated on logits corresponding to all but the last token,
            # predicting the input_ids from the second token onwards.
            labels = input_ids.clone()
            # Set padding token IDs in labels to -100 so they are ignored by the loss function
            labels[labels == tokenizer.pad_token_id] = -100 
            
            optimizer.zero_grad()
            
            # Get probabilities from the frozen base model
            with torch.no_grad(): # Ensure no gradients are computed for base_model
                outputs_base = base_model(input_ids=input_ids, attention_mask=attention_mask)
                # Logits are for the *next* token, so shape is (batch, seq_len, vocab_size)
                probs_base = torch.softmax(outputs_base.logits, dim=-1) 
            
            # Get probabilities from the reweighting model
            outputs_reweighting = reweighting_model(input_ids=input_ids, attention_mask=attention_mask)
            probs_reweighting = torch.softmax(outputs_reweighting.logits, dim=-1)
            
            # Combine probabilities (element-wise product and normalize)
            # p_i = (b_i * r_i) / ||b_i * r_i||_1
            combined_probs_product = probs_base * probs_reweighting
            
            epsilon = 1e-9 # To prevent division by zero or log(0)
            norm_factor = torch.sum(combined_probs_product, dim=-1, keepdim=True) + epsilon
            combined_probs_normalized = combined_probs_product / norm_factor
            
            # Convert combined probabilities back to logits for CrossEntropyLoss
            # CrossEntropyLoss expects raw logits, not probabilities.
            # log(p) is a common way to get logits back from probabilities.
            final_logits_for_loss = torch.log(combined_probs_normalized + epsilon) # Add epsilon for numerical stability

            # Calculate loss
            # Reshape logits to (batch_size * (seq_len-1), vocab_size)
            # Reshape labels to (batch_size * (seq_len-1))
            # We don't predict a token for the very first input token, hence [:, :-1, :]
            # And labels are shifted, so we predict labels[:, 1:]
            loss = criterion(
                final_logits_for_loss[:, :-1, :].contiguous().view(-1, vocab_size),
                labels[:, 1:].contiguous().view(-1)
            )
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward() # Compute gradients only for reweighting_model
                optimizer.step() # Update reweighting_model parameters
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