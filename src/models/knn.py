from sklearn.neighbors import KNeighborsClassifier


def get_model() -> tuple:

    estimator = KNeighborsClassifier()
    param_grid = {"n_neighbors": [3, 5, 7, 9]}
    return estimator, param_grid