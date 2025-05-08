# llm_survey_steering/evaluation/evaluator.py

import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm # Standard tqdm

# Imports from within the package
from ..generation.generator import generate_survey_response
from .metrics import parse_survey_response_value, jensen_shannon_divergence

def evaluate_survey_responses(config_obj, tokenizer, base_model, 
                              reweighting_model, ground_truth_distributions_map):
    """
    Evaluates survey response generation for base and plugin models against ground truth.
    """
    print("\n--- Evaluating Survey Response Generation ---")
    results = []

    if not config_obj.GENERATION_PROMPTS_SURVEY:
        print("No generation prompts provided. Skipping evaluation.")
        return pd.DataFrame()

    # Ensure reweighting_model is passed, even if it's None (for base_only evaluation)
    # The generate_survey_response function handles a None reweighting_model.

    for prompt_text in tqdm(config_obj.GENERATION_PROMPTS_SURVEY, desc="Evaluating Prompts"):
        base_responses_raw = []
        plugin_responses_raw = [] # For model with reweighting
        
        base_responses_parsed = []
        plugin_responses_parsed = []

        # Generate for Base Model
        for _ in range(config_obj.NUM_PREDICTIONS_PER_PROMPT_EVAL):
            raw_base = generate_survey_response(
                prompt_text=prompt_text,
                base_model=base_model,
                tokenizer=tokenizer,
                config_obj=config_obj,
                reweighting_model=None # Explicitly no reweighting model for base
            )
            base_responses_raw.append(raw_base)
            parsed_val = parse_survey_response_value(raw_base)
            if parsed_val is not None and 1 <= parsed_val <= 10:
                base_responses_parsed.append(parsed_val)

        # Generate for Plugin Model (Base + Reweighting)
        # Only if reweighting_model is available and alpha > 0
        if reweighting_model and config_obj.REWEIGHTING_STRENGTH_ALPHA > 0:
            for _ in range(config_obj.NUM_PREDICTIONS_PER_PROMPT_EVAL):
                raw_plugin = generate_survey_response(
                    prompt_text=prompt_text,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    config_obj=config_obj,
                    reweighting_model=reweighting_model
                )
                plugin_responses_raw.append(raw_plugin)
                parsed_val = parse_survey_response_value(raw_plugin)
                if parsed_val is not None and 1 <= parsed_val <= 10:
                    plugin_responses_parsed.append(parsed_val)
        else: # If no reweighting, plugin responses are effectively same as base or empty
            plugin_responses_parsed = [] # Or copy base if that's the desired comparison

        # Create distributions from parsed valid responses
        base_dist_counts = Counter(base_responses_parsed)
        plugin_dist_counts = Counter(plugin_responses_parsed)

        # Normalize generated distributions for JS divergence
        total_base_valid = sum(base_dist_counts.values())
        norm_base_dist_probs = {k: v / total_base_valid for k, v in base_dist_counts.items()} if total_base_valid > 0 else {}
        
        total_plugin_valid = sum(plugin_dist_counts.values())
        norm_plugin_dist_probs = {k: v / total_plugin_valid for k, v in plugin_dist_counts.items()} if total_plugin_valid > 0 else {}

        # Get ground truth distribution for the current prompt
        # ground_truth_distributions_map should already contain probabilities
        ground_truth_dist_for_prompt = ground_truth_distributions_map.get(prompt_text, {})

        # Calculate JS Divergence
        js_base_vs_gt = None
        if ground_truth_dist_for_prompt and norm_base_dist_probs:
            js_base_vs_gt = jensen_shannon_divergence(norm_base_dist_probs, ground_truth_dist_for_prompt)
        elif not ground_truth_dist_for_prompt:
             print(f"Warning: No ground truth for prompt: {prompt_text}")
        
        js_plugin_vs_gt = None
        if ground_truth_dist_for_prompt and norm_plugin_dist_probs:
            js_plugin_vs_gt = jensen_shannon_divergence(norm_plugin_dist_probs, ground_truth_dist_for_prompt)
        elif not norm_plugin_dist_probs and reweighting_model: # Only warn if plugin was expected
             print(f"Warning: No valid plugin responses to compare for prompt: {prompt_text}")


        # Calculate mean responses from valid parsed numbers
        mean_base_response = np.mean(base_responses_parsed) if base_responses_parsed else None
        mean_plugin_response = np.mean(plugin_responses_parsed) if plugin_responses_parsed else None
        
        # Calculate ground truth mean (weighted average from probabilities)
        mean_gt_response = None
        if ground_truth_dist_for_prompt:
            mean_gt_response = sum(k * v for k, v in ground_truth_dist_for_prompt.items())
            if not np.isclose(sum(ground_truth_dist_for_prompt.values()), 1.0) and sum(ground_truth_dist_for_prompt.values()) != 0:
                 print(f"Warning: Ground truth probabilities for prompt '{prompt_text}' do not sum to 1 (sum={sum(ground_truth_dist_for_prompt.values())}). Mean may be affected.")


        results.append({
            "prompt": prompt_text,
            "base_mean_response": mean_base_response,
            "plugin_mean_response": mean_plugin_response,
            "ground_truth_mean_response": mean_gt_response,
            "base_response_distribution_counts": dict(base_dist_counts), # Store as dict
            "plugin_response_distribution_counts": dict(plugin_dist_counts), # Store as dict
            "base_response_distribution_probs": dict(norm_base_dist_probs), # Store as dict
            "plugin_response_distribution_probs": dict(norm_plugin_dist_probs), # Store as dict
            "ground_truth_distribution_probs": dict(ground_truth_dist_for_prompt), # Store as dict
            "js_base_vs_gt": js_base_vs_gt,
            "js_plugin_vs_gt": js_plugin_vs_gt,
            "num_valid_base": len(base_responses_parsed),
            "num_valid_plugin": len(plugin_responses_parsed),
            "raw_base_responses_sample": base_responses_raw[:min(3, len(base_responses_raw))], # Store a few raw samples
            "raw_plugin_responses_sample": plugin_responses_raw[:min(3, len(plugin_responses_raw))]
        })
        
    return pd.DataFrame(results)
