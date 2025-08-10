from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# == Train / model control ==
# Model selection - choose which model to train
MODEL_TYPE = "xgboost"  # Options: "xgboost", "lightgbm", "random_forest", "logistic_regression"

# Escolha quais blocos de features usar no treino.
TRAIN_ON = {
    "imu": True,        # a maioria das features que começam com acc_, rot_, linacc_, acc_mag
    "thm": False,       # thermopile features (thm_)
    "tof": False,       # time-of-flight derived features (tof*)
    "demo": True,       # demographics (age, sex, handedness, etc.)
    "seq_length": True  # sequência length feature
}

# Cross-validation & random seed
N_SPLITS = 5
RANDOM_STATE = 42

# Model parameters - organized by model type
MODEL_PARAMS = {
    "xgboost": {
        "objective": "multi:softprob",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "random_state": RANDOM_STATE,
    },
    "lightgbm": {
        "objective": "multiclass",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "n_estimators": 1000,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
}

# Legacy parameters for backward compatibility
XGB_PARAMS = MODEL_PARAMS["xgboost"]
LGB_PARAMS = MODEL_PARAMS["lightgbm"]

# Early stopping rounds used when fitting with eval_set (only for gradient boosting models)
EARLY_STOPPING_ROUNDS = 50

# Se quiser salvar a lista de colunas do modelo para uso no inference step
SAVE_FEATURE_LIST = True

# Feature selection parameters (mantive aqui caso precise)
N_BOOTSTRAPS = 30
N_LASSO_FEATS = 50
LASSO_C = 0.1
N_SFS_FEATS = 20
SFS_CV_FOLDS = 3



# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
