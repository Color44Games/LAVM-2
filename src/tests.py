import numpy as np
from perceptron import Perceptron
from utilities import *

def run_experiment(n_features: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, lr: float, batch_size: int, init_type: str, exp_name: str = "") -> float:
    model = Perceptron(
        n_features,
        lr,
        100,
        batch_size,
        init_type,
    )

    model.fit(X_train, y_train, X_test, y_test)

    y_predict = model.predict(X_test)
    res_metrics = metrics(y_test, y_predict)

    loss_img_path = f"graphics/loss_{exp_name}.jpg"
    boundary_img_path = f"graphics/boundary_{exp_name}.jpg"

    draw_loss_curves(model.loss_history_train, model.loss_history_test, save_path=loss_img_path)
    draw_decision_boundary(model, X_test, y_test, title=f"Граница: {exp_name}", save_path=boundary_img_path)

    return res_metrics["accuracy"]

