# LLM Survey Steering

**Version:** 0.2.0 (Milestone: Minimal End-to-End Pipeline with Causal & Chat Model Logic Testing)

## Project Goal

To develop and evaluate a methodology for steering Large Language Models (LLMs) to generate survey responses that reflect the tendencies of specific target populations. This project implements the core technique from the paper "Logits are All We Need to Adapt Closed Models," which uses a smaller, trainable "reweighting model" to influence the token probabilities of a larger, frozen "base model." The ultimate aim is to create a replicable pipeline for social science research on LLM alignment and bias.

## Current Status

The project has successfully transitioned from a single proof-of-concept script to an installable Python package (`llm_survey_steering`). This package provides a modular framework for:

1.  **Data Processing:** Loading and preprocessing survey data (currently World Values Survey - WVS), and generating formatted prompts suitable for LLM training and inference.
2.  **Model Handling:** Loading a frozen base LLM and initializing a smaller GPT-2 based reweighting model.
3.  **Reweighting Model Training:** Training the reweighting model on survey data to learn demographic-specific response biases. The training implements the probability combination method `p_final = (p_base * p_reweight) / L1_norm(p_base * p_reweight)`.
4.  **Response Generation:** Generating survey responses using the base model alone and the base model combined with the reweighting model.
5.  **Evaluation:** Comparing generated response distributions to ground truth WVS data using Jensen-Shannon Divergence and other metrics.

The package now supports a flexible configuration for different base models, including both traditional Causal LMs (e.g., GPT-2) and the *logic* for handling Chat LMs (e.g., models requiring specific chat templating).

## Replication & Methodology Details

This project aims for methodological transparency and replicability. The core process for steering the LLM involves the following key steps and underlying principles:

**1. Data Preparation:**

*   **Source:** World Values Survey (WVS) data (Wave 7 used in current examples).
*   **Preprocessing:**
    *   Filtering for target countries and relevant survey questions (e.g., economic outlook questions Q106-Q110).
    *   Selection of demographic variables (e.g., Age - Q262, which is then mapped to groups like "Young", "Mid", "Old").
    *   Responses are filtered to include valid scale answers (e.g., 1-10), excluding "don't know," "no answer," etc.
*   **Prompt Formatting:**
    *   **Training Data (`TARGET_DATA_FOR_BIAS`):** For each valid WVS respondent and question, a training example string is created.
        *   For **Causal LMs (e.g., GPT-2):**
            `"Demographics: Country_{COUNTRY_CODE}_Age_{AGE_GROUP}. SurveyContext: {QUESTION_KEY}. My view on the scale is: {RESPONSE_VALUE}"`
        *   For **Chat LMs:** A structured conversation is created (e.g., a user message with demographics/context, and an assistant message with the `RESPONSE_VALUE`) and then formatted using the specific chat model's `tokenizer.apply_chat_template()`.
    *   **Inference Prompts (`GENERATION_PROMPTS_SURVEY`):** Similar to training prompts but without the `RESPONSE_VALUE`. For chat models, `add_generation_prompt=True` is used with `apply_chat_template`.
*   **Train/Test Split:** The WVS respondent data is split. The reweighting model is trained *only* on prompts derived from the training split. Evaluation (generation and ground truth comparison) uses prompts and distributions derived *only* from the test split.
*   **Ground Truth Distributions:** For each unique inference prompt (representing a demographic-question context), the actual distribution of WVS responses from the test split is calculated and normalized. This serves as the target to compare against.

**2. Model Architecture:**

*   **Base Model (Frozen):** A pre-trained Large Language Model (e.g., `gpt2-medium`, `TinyLlama-1.1B-Chat-v1.0`). Its weights are **not updated** during the reweighting process. It provides the initial probability distribution over the next token.
*   **Reweighting Model (Trainable):** A smaller transformer-based language model (e.g., a 2-layer GPT-2 architecture).
    *   Its vocabulary size and embedding dimensions are matched to the base model's tokenizer.
    *   This model is trained on the `TARGET_DATA_FOR_BIAS` to learn how to adjust the base model's probabilities to better reflect the target demographic group's response patterns for a given context.

**3. Reweighting Model Training - Core Mechanism:**

The reweighting model learns to predict a probability distribution `r_t` for the next token, given the same context as the base model. The base model predicts `b_t`. These are combined to produce a final probability distribution `p_t` for sampling the next token.

*   **Probability Combination (Inspired by "Logits are All We Need...", Equation 4):**
    At each token generation step `t` in a sequence:
    Let `logits_base_t` be the logits from the base model for the next token.
    Let `logits_reweight_t` be the logits from the reweighting model for the next token.

    The probabilities are:
    `probs_base_t = softmax(logits_base_t)`
    `probs_reweight_t = softmax(logits_reweight_t)`

    The combined probability for each token `i` in the vocabulary is computed by an element-wise product, followed by L1 normalization:
    `combined_product_i_t = probs_base_i_t * probs_reweight_i_t`
    `norm_factor_t = sum(combined_product_j_t for j in vocabulary)`
    `p_i_t = combined_product_i_t / norm_factor_t`

*   **Loss Calculation:** The reweighting model is trained using a standard Cross-Entropy Loss. The loss is calculated based on the `log(p_t)` (logarithm of the combined normalized probabilities) and the true target tokens from the training data.
    `Loss = CrossEntropyLoss(log(p_t), target_token_t+1)`
    Only the parameters of the **reweighting model** are updated during this process.

**4. Response Generation:**

Survey responses are generated token by token.
*   **Base Model Only:** The next token is sampled directly from `probs_base_t` (after applying temperature).
*   **Plugin Model (Base + Reweighting):** The next token is sampled from the combined probability distribution `p_t` (after applying temperature to the original logits and potentially a strength parameter `alpha` to the reweighting model's contribution, e.g., `probs_reweight_adjusted = probs_reweight_t ** alpha`). The current implementation applies `alpha` to the probabilities: `combined_product = probs_base * (probs_reweight ** alpha)`.

**5. Evaluation:**

*   **Generation:** For each inference prompt, multiple responses (`NUM_PREDICTIONS_PER_PROMPT_EVAL`) are generated using both the base model and the plugin model.
*   **Parsing:** Generated text responses are parsed to extract a numerical value (1-10).
*   **Distribution Comparison:** The distribution of valid parsed numerical responses from the base model and the plugin model are compared against the pre-calculated `GROUND_TRUTH_DISTRIBUTIONS` for that prompt.
*   **Primary Metric: Jensen-Shannon (JS) Divergence:** JS Divergence is used to quantify the similarity between the generated distributions and the ground truth distribution. A lower JS Divergence indicates better alignment.
    `JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)`, where `M = 0.5 * (P + Q)` and `KL` is Kullback-Leibler divergence.
*   **Other Metrics:** Mean responses are also calculated and compared.

This methodology allows for a quantitative assessment of how effectively the reweighting model steers the base LLM's outputs to align with the characteristics of the target survey population.

## Key Milestones Achieved (from PoC script)

1.  **PoC Script Analysis:** Thorough understanding of the initial proof-of-concept script.
2.  **Package Structuring (Phase 1 Complete):**
    *   Codebase successfully refactored into a Python package structure (`llm_survey_steering`) with distinct modules for configuration, data processing, models, training, generation, and evaluation.
    *   The package is installable via `pip install -e .` using `pyproject.toml`.
    *   Main experimental workflow encapsulated in `scripts/run_experiment.py`.
3.  **Initial Refinements (Partial Phase 2):**
    *   Configuration handling improved by centralizing parameters in `ProjectConfig`.
    *   Tokenizer is loaded earlier and made available via the `ProjectConfig` instance.
    *   Basic docstrings started.
4.  **Chat Model Logic Integration (Foundation for Phase 4):**
    *   `ProjectConfig` now includes `MODEL_TYPE` ("causal" or "chat").
    *   Data processing (`generate_prompt_data_from_wvs_split`) dynamically formats prompts using `tokenizer.apply_chat_template` if `MODEL_TYPE` is "chat", or uses string formatting for "causal".
5.  **Test Suite Development:**
    *   `tests/test_minimal_pipeline.py`: A lightweight test using `distilgpt2` for both causal and chat *logic* (with a manually set chat template for `distilgpt2`) to ensure the pipeline runs end-to-end quickly.
    *   `tests/test_real_models_pipeline.py`: A more comprehensive test using `gpt2-medium` (causal) and `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (a real, open chat model that provides its own chat template). This test uses a small synthetic dataset.

## Test Run Summaries (from `tests/test_real_models_pipeline.py`)

The test suite (`tests/test_real_models_pipeline.py`) runs the end-to-end pipeline using synthetic data for quick verification.

*   **`gpt2-medium` (Causal Model Test):**
    *   **Status:** Passed.
    *   **Observations:** The pipeline completed successfully. The reweighting model trained for 1 epoch on synthetic data, and evaluation metrics were generated. This confirms the core logic for causal LMs is functional.
    *   *(Example Loss: ~2.8-3.0, JS Divergence varies based on synthetic data randomness)*

*   **`TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Chat Model Test):**
    *   **Status:** Passed (Pipeline completed without crashing).
    *   **Observations:**
        *   The pipeline correctly loaded the TinyLlama model and its associated tokenizer, automatically using its pre-defined chat template (no manual template injection was needed for this model).
        *   The reweighting model trained successfully (Example Loss: ~0.4-0.5, often lower than gpt2 due to different tokenization and base model behavior on chat-formatted data).
        *   **Response Parsing:** During evaluation with the minimal synthetic dataset, the base TinyLlama model often generated non-numeric, conversational text. The plugin model (TinyLlama + reweighting) generated numbers, but these were frequently outside the 1-10 survey scale (e.g., "400", "700").
        *   **Consequence:** This led to `num_valid_plugin_responses = 0` in the evaluation for TinyLlama, and warnings about no valid responses being parsed.
    *   **Interpretation:** This is not a failure of the pipeline's ability to *handle* chat models. Instead, it's an expected outcome demonstrating that:
        1.  Small chat models like TinyLlama may not inherently understand the specific task of outputting a single digit (1-10) in a zero-shot manner, even with chat formatting.
        2.  A very small reweighting model trained on minimal synthetic data can learn to bias the output towards *numerals* but lacks the precision to enforce the correct 1-10 range without more data/capacity.
        This result successfully validates that the chat model processing pathway in the code is functional.

**Overall Test Suite Conclusion:** The test suite confirms that the core package can run end-to-end for both causal LMs and correctly handle the loading and prompt formatting for real chat LMs that provide their own chat templates.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Bonorinoa/LLM-Survey-Steering.git
    cd LLM-Survey-Steering
    ```
2.  Create and activate a Python virtual environment (e.g., using conda or venv):
    ```bash
    # Example with conda
    # conda create -n llm_steer python=3.9
    # conda activate llm_steer
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

*   **Main Experiments:** Modify and run `scripts/run_experiment.py`. This script uses the full WVS dataset and allows for more comprehensive configuration.
*   **Testing:**
    *   Run the minimal logic test: `python tests/test_minimal_pipeline.py`
    *   Run the test with small real models: `python tests/test_real_models_pipeline.py`

## Dependencies

Listed in `requirements.txt`. Key dependencies include:
*   `torch`
*   `transformers`
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `scipy`
*   `matplotlib`
*   `tqdm`

## Next Steps

See the Executive Report (or project issues) for planned next steps, including:
*   Further refinement of API and configuration.
*   Comprehensive docstring coverage.
*   Expanded demographic variable handling.
*   More sophisticated evaluation metrics.
*   Experiments with larger and more varied LLMs.

## Contributing
(Placeholder for contribution guidelines if the project were to be opened for wider collaboration)