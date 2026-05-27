import numpy as np
import matplotlib.pyplot as plt

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0 
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy" : float(accuracy),
        "precision" : float(precision),
        "recall" : float(recall),
        "f1_score" : float(f1_score),
        "confusion_matrix" : {"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)}
    }

def draw_loss_curves(train_losses: list[float], val_losses: list[float], save_path: str | None = None):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train", color="blue", lw=2)
    
    if val_losses and len(val_losses) > 0:
        plt.plot(val_losses, label="Validation", color="orange", lw=2, linestyle="--")
        
    plt.title("Динамика функции потерь (BCE) по эпохам", fontsize=14)
    plt.xlabel("Эпоха", fontsize=12)
    plt.ylabel("Значение потерь", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

def draw_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "Разделяющая поверхность", save_path: str | None = None):
    plt.figure(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    probs = model.predict_probability(grid_points)
    probs = probs.reshape(xx.shape)
    
    contour = plt.contourf(xx, yy, probs, levels=25, cmap="RdBu_r", alpha=0.3)
    plt.colorbar(contour, label="Вероятность класса 1")
    
    plt.contour(xx, yy, probs, levels=[0.5], colors="green", linewidths=3, linestyles="-")
    
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", edgecolor="k", label="Класс 0", alpha=0.8, s=50)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", edgecolor="k", label="Класс 1", alpha=0.8, s=50)
    
    plt.title(title, fontsize=14)
    plt.xlabel("X1", fontsize=12)
    plt.ylabel("X2", fontsize=12)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.4)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()