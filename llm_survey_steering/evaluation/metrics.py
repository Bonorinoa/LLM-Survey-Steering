# llm_survey_steering/evaluation/metrics.py

import numpy as np
import re
from scipy.stats import entropy # Using scipy for now as in original

def jensen_shannon_divergence(p, q, base=2):
    """
    Calculates Jensen-Shannon divergence between two probability distributions.
    Assumes p and q are dictionaries mapping outcomes to probabilities.
    Converts dicts to numpy arrays aligned by sorted keys for scipy.stats.entropy.
    """
    # Get all unique outcomes (keys) from both distributions
    all_outcomes = sorted(list(set(list(p.keys()) + list(q.keys()))))

    # Create probability vectors aligned with all_outcomes
    # If an outcome is not in a distribution, its probability is 0
    p_vec = np.array([p.get(outcome, 0) for outcome in all_outcomes])
    q_vec = np.array([q.get(outcome, 0) for outcome in all_outcomes])

    # Normalize vectors to ensure they are valid probability distributions
    # (sum to 1). This handles cases where input dicts might not be perfectly normalized
    # or if they represent counts that need normalization.
    if np.sum(p_vec) > 0:
        p_vec = p_vec / np.sum(p_vec)
    else: # Avoid division by zero if p_vec is all zeros
        p_vec = np.zeros_like(p_vec) # Or handle as error/specific value

    if np.sum(q_vec) > 0:
        q_vec = q_vec / np.sum(q_vec)
    else: # Avoid division by zero if q_vec is all zeros
        q_vec = np.zeros_like(q_vec)


    # Add a small epsilon to avoid log(0) issues if one distribution has zero
    # probability for an outcome where the other does not, AFTER initial normalization.
    # This step is more critical if calculating KL divergence directly.
    # Scipy's entropy might handle this, but being explicit can be safer.
    # However, the original script added epsilon *before* normalization.
    # Let's follow the original script's JSD which re-normalized after epsilon.
    
    # Re-implementing the original logic for adding epsilon and re-normalizing:
    epsilon = 1e-9
    p_vec_eps = np.where(p_vec == 0, epsilon, p_vec)
    q_vec_eps = np.where(q_vec == 0, epsilon, q_vec)
    
    # Re-normalize after adding epsilon
    p_vec_eps_norm = p_vec_eps / np.sum(p_vec_eps)
    q_vec_eps_norm = q_vec_eps / np.sum(q_vec_eps)

    m_vec = 0.5 * (p_vec_eps_norm + q_vec_eps_norm)

    # Calculate KL divergences for JSD formula
    # entropy(pk, qk) computes D_KL(pk || qk)
    kl_p_m = entropy(p_vec_eps_norm, m_vec, base=base)
    kl_q_m = entropy(q_vec_eps_norm, m_vec, base=base)
    
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    # JSD should be between 0 and 1 (for base 2) or 0 and ln(2) (for base e)
    # Handle potential floating point inaccuracies leading to tiny negative values or values slightly above max
    max_jsd = 1.0 if base == 2 else np.log(2)
    jsd = np.clip(jsd, 0, max_jsd)

    return jsd if np.isfinite(jsd) else max_jsd # Return max divergence if issues

def parse_survey_response_value(text_response):
    """
    Parses a numerical survey response (1-10) from a text string.
    Returns an integer if found, otherwise None.
    """
    if text_response is None:
        return None
    try:
        # Find all sequences of digits
        numbers = re.findall(r'\d+', text_response)
        if numbers:
            # Convert the first found number to an integer
            num = int(numbers[0])
            # Check if the number is within the typical survey scale (1-10)
            # if 1 <= num <= 10: # This check is done later in evaluation loop
            return num
    except ValueError: # Handles cases where conversion to int might fail unexpectedly
        pass
    except Exception as e: # Catch any other unexpected errors during parsing
        print(f"Unexpected error parsing response '{text_response}': {e}")
    return None
