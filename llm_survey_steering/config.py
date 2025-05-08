# llm_survey_steering/config.py

import torch
import numpy as np

# --- Configuration for Data Processing (can be adjusted) ---
WVS_FILE_PATH = 'llm_survey_steering/data/WVS_Cross-National_Wave_7_csv_v6_0.csv' # Adjusted path relative to project root

GENERAL_VARIABLES = ['B_COUNTRY_ALPHA']
ECONOMIC_QUESTION_IDS = [f'Q{i}' for i in range(106, 111)] # Q106 to Q110
DEMOGRAPHIC_VARIABLES_IDS = ['Q262'] # Example: Age (Q260=Sex, Q275=Education)

ECON_QUESTIONS_MAP = {
    "Q106": {"key": "ECON_IncomeEqualityVsIndividualEffort", "full_text": "On a scale from 1 to 10, where 1 means 'Incomes should be made more equal' and 10 means 'There should be greater incentives for individual effort'"},
    "Q107": {"key": "ECON_PrivateVsGovOwnership", "full_text": "On a scale from 1 to 10, where 1 means 'Private ownership should be increased' and 10 means 'Government ownership should be increased'"},
    "Q108": {"key": "ECON_GovVsPeopleResponsibility", "full_text": "On a scale from 1 to 10, where 1 means 'Government should take more responsibility' and 10 means 'People should take more responsibility'"},
    "Q109": {"key": "ECON_CompetitionGoodVsHarmful", "full_text": "On a scale from 1 to 10, where 1 means 'Competition is good' and 10 means 'Competition is harmful'"},
    "Q110": {"key": "ECON_HardWorkVsLuck", "full_text": "On a scale from 1 to 10, where 1 means 'Hard work usually brings a better life' and 10 means 'Hard work doesn’t generally bring success'"},
}

TARGET_COUNTRIES = ['ARG', 'USA', 'NGA', 'JPN'] # Example countries
TEST_SET_SIZE = 0.60 # 60% of the WVS subset for testing, 40% for training reweighting model
RANDOM_STATE_SPLIT = 42 # For reproducible train-test split

# --- Prompt Formats ---
# For traditional Causal LMs
CAUSAL_LM_TRAINING_PROMPT_FORMAT = "Demographics: Country_{}_Age_{}. SurveyContext: {}. My view on the scale is: {}"
CAUSAL_LM_INFERENCE_PROMPT_FORMAT = "Demographics: Country_{}_Age_{}. SurveyContext: {}. My view on the scale is:"

# Chat model prompts will be handled by functions using tokenizer.apply_chat_template

class ProjectConfig:
    """
    Configuration class for the LLM Survey Steering project.

    This class centralizes all parameters required for data processing,
    model setup, training, generation, and evaluation.
    """
    def __init__(self):
        # --- Data related paths and definitions ---
        self.WVS_FILE_PATH: str = WVS_FILE_PATH
        self.GENERAL_VARIABLES: list[str] = GENERAL_VARIABLES
        self.ECONOMIC_QUESTION_IDS: list[str] = ECONOMIC_QUESTION_IDS
        self.DEMOGRAPHIC_VARIABLES_IDS: list[str] = DEMOGRAPHIC_VARIABLES_IDS
        self.ECON_QUESTIONS_MAP: dict = ECON_QUESTIONS_MAP
        self.TARGET_COUNTRIES: list[str] = TARGET_COUNTRIES
        self.TEST_SET_SIZE: float = TEST_SET_SIZE
        self.RANDOM_STATE_SPLIT: int = RANDOM_STATE_SPLIT

        # --- Prompt Formats (for Causal LMs, chat models use specific functions) ---
        self.CAUSAL_LM_TRAINING_PROMPT_FORMAT: str = CAUSAL_LM_TRAINING_PROMPT_FORMAT
        self.CAUSAL_LM_INFERENCE_PROMPT_FORMAT: str = CAUSAL_LM_INFERENCE_PROMPT_FORMAT

        # --- Model & Training Hyperparameters ---
        self.BASE_MODEL_NAME: str = "gpt2-medium" # Can be changed to a chat model like "meta-llama/Meta-Llama-3-8B-Instruct"
        self.MODEL_TYPE: str = "causal" # "causal" or "chat". Determines prompt formatting.
                                        # If BASE_MODEL_NAME contains "instruct" or "chat", consider setting this to "chat".

        self.REWEIGHTING_MODEL_N_LAYER: int = 2
        self.REWEIGHTING_MODEL_N_HEAD: int = 4
        self.LEARNING_RATE: float = 3e-5
        self.EPOCHS: int = 10
        self.BATCH_SIZE: int = 4
        self.MAX_SEQ_LENGTH: int = 128
        
        # --- Generation Parameters ---
        self.GENERATION_MAX_LEN_SURVEY: int = 3
        self.GENERATION_TEMPERATURE_BASE: float = 0.7
        self.GENERATION_TEMPERATURE_REWEIGHT: float = 0.7
        self.REWEIGHTING_STRENGTH_ALPHA: float = 1.0
        self.NUM_PREDICTIONS_PER_PROMPT_EVAL: int = 20
        
        # --- Reproducibility ---
        self.RANDOM_SEED: int = 42

        # --- Dynamic attributes (populated during runtime) ---
        self.TARGET_DATA_FOR_BIAS: list[str] = []
        self.GENERATION_PROMPTS_SURVEY: list[str] = []
        self.GROUND_TRUTH_DISTRIBUTIONS: dict = {} # Maps prompt to its ground truth response distribution
        
        self.tokenizer = None # Will be populated by load_models_and_tokenizer
        self.vocab_size = None # Will be populated by load_models_and_tokenizer

        # --- Device Setup ---
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_seeds(self):
        """Sets random seeds for numpy and torch for reproducibility."""
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.RANDOM_SEED)

    def _determine_model_type_if_needed(self):
        """
        Automatically sets self.MODEL_TYPE to 'chat' if 'auto' and
        BASE_MODEL_NAME suggests a chat/instruct model.
        This is a basic heuristic.
        """
        if self.MODEL_TYPE == "auto": # If user wants auto-detection
            if any(keyword in self.BASE_MODEL_NAME.lower() for keyword in ["instruct", "chat", "llama-3", "mixtral"]):
                print(f"Auto-detected chat model based on name: {self.BASE_MODEL_NAME}. Setting MODEL_TYPE to 'chat'.")
                self.MODEL_TYPE = "chat"
            else:
                print(f"Auto-detection did not identify {self.BASE_MODEL_NAME} as chat model. Defaulting MODEL_TYPE to 'causal'.")
                self.MODEL_TYPE = "causal"
        elif self.MODEL_TYPE not in ["causal", "chat"]:
            print(f"Warning: Invalid MODEL_TYPE '{self.MODEL_TYPE}'. Defaulting to 'causal'.")
            self.MODEL_TYPE = "causal"


# Example of how to change config for a chat (needs huggingface cli login to authentica) model:
# config = ProjectConfig()
# config.BASE_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1" # or any other chat model
# config.MODEL_TYPE = "chat" # Explicitly set, or use "auto" for _determine_model_type_if_needed()
# config._determine_model_type_if_needed() # Call if MODEL_TYPE is "auto" or for validation
