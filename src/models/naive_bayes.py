from sklearn.naive_bayes import GaussianNB


def get_model() -> tuple:

    estimator = GaussianNB()
    param_grid = {"var_smoothing": [1e-9, 1e-8, 1e-7]}
    return estimator, param_grid