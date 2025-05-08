# llm_survey_steering/config.py

import torch
import numpy as np

# --- Configuration for Data Processing (can be adjusted) ---
WVS_FILE_PATH = 'data/WVS_Cross-National_Wave_7_csv_v6_0.csv' # Adjusted path relative to project root

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

TRAINING_PROMPT_FORMAT = "Demographics: Country_{}_Age_{}. SurveyContext: {}. My view on the scale is: {}"
INFERENCE_PROMPT_FORMAT = "Demographics: Country_{}_Age_{}. SurveyContext: {}. My view on the scale is:"


# --- Global Config for Models and Training ---
class ProjectConfig:
    def __init__(self):
        # --- Data related paths and definitions (moved from top-level for consolidation if needed) ---
        self.WVS_FILE_PATH = WVS_FILE_PATH
        self.GENERAL_VARIABLES = GENERAL_VARIABLES
        self.ECONOMIC_QUESTION_IDS = ECONOMIC_QUESTION_IDS
        self.DEMOGRAPHIC_VARIABLES_IDS = DEMOGRAPHIC_VARIABLES_IDS
        self.ECON_QUESTIONS_MAP = ECON_QUESTIONS_MAP
        self.TARGET_COUNTRIES = TARGET_COUNTRIES
        self.TEST_SET_SIZE = TEST_SET_SIZE
        self.RANDOM_STATE_SPLIT = RANDOM_STATE_SPLIT
        self.TRAINING_PROMPT_FORMAT = TRAINING_PROMPT_FORMAT
        self.INFERENCE_PROMPT_FORMAT = INFERENCE_PROMPT_FORMAT

        # --- Model & Training Hyperparameters ---
        self.BASE_MODEL_NAME = "gpt2-medium"
        self.REWEIGHTING_MODEL_N_LAYER = 2
        self.REWEIGHTING_MODEL_N_HEAD = 4
        self.LEARNING_RATE = 3e-5
        self.EPOCHS = 10 # Reduced for quicker runs, adjust as needed
        self.BATCH_SIZE = 4
        self.MAX_SEQ_LENGTH = 128 # Increased for longer demographic + question prompts
        
        # --- Generation Parameters ---
        self.GENERATION_MAX_LEN_SURVEY = 3 # For " 10" or " 7" etc. + EOS
        self.GENERATION_TEMPERATURE_BASE = 0.7
        self.GENERATION_TEMPERATURE_REWEIGHT = 0.7
        self.REWEIGHTING_STRENGTH_ALPHA = 1.0
        self.NUM_PREDICTIONS_PER_PROMPT_EVAL = 20 # Increased for better distribution estimation
        
        # --- Reproducibility ---
        self.RANDOM_SEED = 42

        # --- Dynamic attributes (will be populated during runtime) ---
        self.TARGET_DATA_FOR_BIAS = []
        self.GENERATION_PROMPTS_SURVEY = []
        self.GROUND_TRUTH_DISTRIBUTIONS = {} # Maps prompt to its ground truth response distribution

        # --- Device Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_seeds(self):
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.RANDOM_SEED)

# To be instantiated in the main script
# project_config = ProjectConfig()
# project_config.setup_seeds()
# print(f"Using device: {project_config.device}")