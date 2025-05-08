# llm_survey_steering/evaluation/plots.py

import matplotlib.pyplot as plt
import numpy as np
# from collections import Counter # Not needed directly if distributions are passed

def plot_loss(loss_history, save_path=None):
    """Plots the training loss history."""
    if not loss_history:
        print("Loss history is empty. Skipping plotting.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Reweighting Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    plt.show()

def plot_response_distributions(results_df, save_path_prefix=None):
    """
    Plots the response distributions for base, plugin, and ground truth
    for each prompt in the results DataFrame.
    """
    print("\n--- Plotting Survey Response Distributions (vs Ground Truth) ---")
    num_prompts = len(results_df)
    if num_prompts == 0:
        print("No evaluation results to plot.")
        return

    # One plot per prompt for clarity with 3 distributions
    # If too many prompts, consider summarizing or paginating plots

    for i, (idx, row_data) in enumerate(results_df.iterrows()):
        plt.figure(figsize=(12, 7)) # Individual figure for each prompt
        ax = plt.gca()

        prompt_full = row_data['prompt']
        # Attempt to create a shorter title for the plot
        try:
            demographics_part = prompt_full.split("Demographics: ")[1].split(" SurveyContext:")[0]
            survey_context_part = prompt_full.split("SurveyContext: ")[1].split(". My view")[0]
            prompt_short = f"{demographics_part}\nQ: {survey_context_part}"
        except IndexError:
            prompt_short = prompt_full[:100] + "..." # Fallback short title


        base_dist_counts = row_data.get('base_response_distribution_counts', {})
        plugin_dist_counts = row_data.get('plugin_response_distribution_counts', {})
        gt_dist_probs = row_data.get('ground_truth_distribution_probs', {})

        # X-axis labels (survey responses 1 to 10)
        x_labels = list(range(1, 11))
        x = np.arange(len(x_labels)) # the label locations

        # Get counts for each label for base and plugin
        base_counts = [base_dist_counts.get(str(label), base_dist_counts.get(label, 0)) for label in x_labels]
        plugin_counts = [plugin_dist_counts.get(str(label), plugin_dist_counts.get(label, 0)) for label in x_labels]


        # For ground truth, we have probabilities. To plot on a comparable y-axis to counts,
        # scale GT probabilities by the number of *plugin* predictions (or base, or average).
        # This is for visual comparison of shape; the JS divergence uses actual probabilities.
        num_plugin_preds_for_scaling = sum(plugin_counts)
        if num_plugin_preds_for_scaling == 0: # If no plugin preds, use base, or default to 1
            num_plugin_preds_for_scaling = sum(base_counts) if sum(base_counts) > 0 else 1
        
        gt_scaled_counts = [gt_dist_probs.get(str(label), gt_dist_probs.get(label, 0)) * num_plugin_preds_for_scaling for label in x_labels]

        width = 0.25  # the width of the bars

        rects1 = ax.bar(x - width, base_counts, width, label=f'Base Model (N={sum(base_counts)})', alpha=0.8)
        rects2 = ax.bar(x, plugin_counts, width, label=f'Plugin Model (N={sum(plugin_counts)})', alpha=0.8)
        rects3 = ax.bar(x + width, gt_scaled_counts, width, label=f'Ground Truth (Scaled from Probs, N_gt_eff={num_plugin_preds_for_scaling:.0f})', alpha=0.6)

        ax.set_ylabel('Frequency / Scaled Frequency')
        ax.set_xlabel('Survey Response Value (1-10)')
        ax.set_title(f"Response Distribution: {prompt_short}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()

        ax.bar_label(rects1, padding=3, fontsize=8)
        ax.bar_label(rects2, padding=3, fontsize=8)
        ax.bar_label(rects3, padding=3, fontsize=8, fmt='%.1f') # Show scaled GT counts

        plt.tight_layout()
        if save_path_prefix:
            # Create a safe filename from the prompt
            safe_prompt_filename = "".join([c if c.isalnum() else "_" for c in prompt_short[:50]])
            plot_filename = f"{save_path_prefix}_dist_prompt_{idx}_{safe_prompt_filename}.png"
            plt.savefig(plot_filename)
            print(f"Distribution plot saved to {plot_filename}")
        plt.show()
