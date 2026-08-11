import numpy as np
from metrics import rmse
from tsk import TSKModel


def train_four_models(X_train, Y_train, X_test, Y_test, model_cfg):
    """
    Train four separate TSK models (one per output).
    """
    output_names = ["stability", "flow", "itsm20", "itsm30"]

    models = {}
    results = {}

    for i, name in enumerate(output_names):
        model = TSKModel(**model_cfg.__dict__)
        model.fit(X_train, Y_train[:, i])

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        results[name] = {
            "train_rmse": rmse(Y_train[:, i], pred_train),
            "test_rmse": rmse(Y_test[:, i], pred_test),
        }

        models[name] = model

    return models, results
