from sklearn.svm import SVC

from config import RANDOM_SEED


def get_model() -> tuple:

    estimator = SVC(probability=True, random_state=RANDOM_SEED)
    param_grid = {"C": [0.1, 1.0, 10.0], "kernel": ["rbf"]}
    return estimator, param_grid