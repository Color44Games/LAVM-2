import numpy as np
import time
from perceptron import Perceptron
from utilities import *

def run_experiment(
        n_features: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lr: float, batch_size: int,
        init_type: str, exp_name: str = "",
        loss_type: str = "bce",
        lambda_reg: float = 0.0,
        draw_roc: bool = False,
        draw_plots: bool = True
) -> dict:

    model = Perceptron(
        n_features,
        lr,
        100,
        batch_size,
        init_type,
        loss_type,
        lambda_reg
    )

    start = time.time()
    model.fit(X_train, y_train, X_test, y_test)
    end = time.time()

    learn_time = end - start
    weight_norm = float(np.linalg.norm(model.w))

    y_predict = model.predict(X_test)
    y_prob = model.predict_probability(X_test)

    res_metrics = metrics(y_test, y_predict, y_prob)
    res_metrics["time"] = learn_time
    res_metrics["weight_norm"] = weight_norm

    if draw_plots:
        loss_img_path = f"graphics/loss_{exp_name}.jpg"
        boundary_img_path = f"graphics/boundary_{exp_name}.jpg"
        
        draw_loss_curves(model.loss_history_train, model.loss_history_test, save_path=loss_img_path)
        draw_decision_boundary(model, X_test, y_test, title=f"Граница: {exp_name}", save_path=boundary_img_path)

        if draw_roc:
            roc_img_path = f"graphics/roc_{exp_name}.jpg"
            draw_roc_curve(res_metrics["fpr"], res_metrics["tpr"], res_metrics["roc_auc"], title=f"ROC: {exp_name}", save_path=roc_img_path)

    return res_metrics