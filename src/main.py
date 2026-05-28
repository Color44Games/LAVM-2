import numpy as np
import matplotlib.pyplot as plt
import os

from tests import run_experiment
from data_prepare import *
from utilities import *

def main() -> None: 
    os.makedirs("graphics", exist_ok= True)

    X, y = generate_linear_gauss(n_samples=500, noise_level=0.1)
    X_train, y_train, X_test, y_test = split_data(X, y, train_part=0.7, random_state=42)
    X_train_scaled, X_test_scaled = standartizate_data(X_train, X_test)

    X_xor, y_xor = generate_xor(n_samples=500, noise_level=0.05)
    X_train_xor, y_train_xor, X_test_xor, y_test_xor = split_data(X_xor, y_xor, train_part=0.7, random_state=42)
    X_train_xor_scaled, X_test_xor_scaled = standartizate_data(X_train_xor, X_test_xor)

    X_circle, y_circle = generate_circle(n_samples=500, noise_level=0.05)
    X_train_circle, y_train_circle, X_test_circle, y_test_circle = split_data(X_circle, y_circle, train_part=0.7, random_state=42)
    X_train_circle_scaled, X_test_circle_scaled = standartizate_data(X_train_circle, X_test_circle)
    
    n_features = X_train_scaled.shape[1]

    results_lr = {}
    results_batch = {}
    results_init = {}
    results_l2 = {}

    lr_variants = [0.001, 0.01, 0.5, 1.0]
    batch_variants = [1, 16, 64, 256]
    lambda_variants = [0.0001, 0.001, 0.05, 0.2]

    _ = run_experiment(
        n_features, X_train_scaled, y_train, X_test_scaled, y_test,
        lr=0.1, batch_size=32, init_type="random", 
        exp_name="roc_test", draw_roc=True, draw_plots=True
    )

    only_metrics = run_experiment(
        n_features, X_train_scaled, y_train, X_test_scaled, y_test,
        lr=0.1, batch_size=32, init_type="random", 
        exp_name="standalone_no_plots", draw_roc=False, draw_plots=False
    )

    for current_lr in lr_variants:
        results_lr[current_lr] = run_experiment(
            n_features, X_train_scaled, y_train, X_test_scaled, y_test,
            current_lr, 32, "random", exp_name=f"lr_{current_lr}"
        )

    for current_batch in batch_variants:
        results_batch[current_batch] = run_experiment(
            n_features, X_train_scaled, y_train, X_test_scaled, y_test,
            0.1, current_batch, "random", exp_name= f"batch_{current_batch}"
        )

    results_init["zeros"] = run_experiment(
        n_features, X_train_scaled, y_train, X_test_scaled, y_test,
        0.1, 32, init_type="zeros", exp_name="init_zeros"
    )
    results_init["random"] = run_experiment(
        n_features, X_train_scaled, y_train, X_test_scaled, y_test,
        0.1, 32, init_type="random", exp_name="init_random"
    )
    results_init["large"] = run_experiment(
        n_features, X_train_scaled, y_train, X_test_scaled, y_test,
        0.1, 32, init_type="large", exp_name="init_large"
    )
    
    metrics_xor = run_experiment(
        n_features=2, 
        X_train=X_train_xor_scaled, y_train=y_train_xor, 
        X_test=X_test_xor_scaled, y_test=y_test_xor, 
        lr=0.1, batch_size=32, init_type="random", 
        exp_name="nonlinear_xor"
    )

    metrics_circle = run_experiment(
        n_features=2, 
        X_train=X_train_circle_scaled, y_train=y_train_circle, 
        X_test=X_test_circle_scaled, y_test=y_test_circle, 
        lr=0.1, batch_size=32, init_type="random", 
        exp_name="nonlinear_circle"
    )

    metrics_bce = run_experiment(
        n_features, X_train_scaled, y_train, X_test_scaled, y_test,
        lr=0.1, batch_size=32, init_type="random", exp_name="bce", loss_type="bce"
    )

    metrics_hinge = run_experiment(
        n_features, X_train_scaled, y_train, X_test_scaled, y_test,
        lr=0.1, batch_size=32, init_type="random", exp_name="hinge", loss_type="hinge_loss"
    )

    for lam in lambda_variants:
        results_l2[lam] = run_experiment(
            n_features, X_train_scaled, y_train, X_test_scaled, y_test, 
            lr=0.1, batch_size=64, init_type="random", 
            exp_name=f"l2_reg_{lam}", loss_type="bce", lambda_reg=lam
        )

    print()
    print("Метрики")
    print("-" * 80)
    print(f"Accuracy:      {only_metrics['accuracy']:.6f}")
    print(f"Precision:     {only_metrics['precision']:.6f}")
    print(f"Recall:        {only_metrics['recall']:.6f}")
    print(f"F1-Score:      {only_metrics['f1_score']:.6f}")
    print(f"ROC AUC:       {only_metrics['roc_auc']:.6f}")
    print("-" * 80)
    
    print()
    print("Таблица 1: Влияние скорости обучения")
    print("-" * 80)
    print("Параметр lr  | Accuracy")
    print("-" * 80)
    for k, v in results_lr.items():
        print(f"{k:<12} | {v['accuracy']:.6f}")
    print("-" * 80)

    print()
    print("Таблица 2: Влияние размера батча (batch_size)")
    print("-" * 80)
    print("Размер батча  | Время обучения (с)")
    print("-" * 80)
    for k, v in results_batch.items():
        print(f"{k:<13} | {v['time']:.6f}")
    print("-" * 80)

    print()
    print("Таблица 3: Влияние инициализации весов")
    print("-" * 80)
    print("Тип инициализации  | Accuracy")
    print("-" * 80)
    for k, v in results_init.items():
        print(f"{k:<18} | {v['accuracy']:.6f}")
    print("-" * 80)

    print()
    print("Таблица 4: Влияние выборки данных")
    print("-" * 80)
    print("Тип выборки | Accuracy | F1-Score | ROC AUC")
    print("-" * 80)
    print(f"{'Линейная':<11} | {results_init['random']['accuracy']:.6f} | {results_init['random']['f1_score']:.6f} | {results_init['random']['roc_auc']:.6f}")
    print(f"{'XOR':<11} | {metrics_xor['accuracy']:.6f} | {metrics_xor['f1_score']:.6f} | {metrics_xor['roc_auc']:.6f}")
    print(f"{'Окружность':<11} | {metrics_circle['accuracy']:.6f} | {metrics_circle['f1_score']:.6f} | {metrics_circle['roc_auc']:.6f}")
    print("-" * 80)
    print()

    print()
    print("Таблица 5: Сравнение BCE Loss и Hinge Loss")
    print("-" * 80)
    print("Функция потерь | Accuracy | Precision | Recall   | F1-Score | ROC AUC")
    print("-" * 80)
    print(f"{'BCE':<14} | {metrics_bce['accuracy']:.6f} | {metrics_bce['precision']:.6f}  | {metrics_bce['recall']:.6f} | {metrics_bce['f1_score']:.6f} | {metrics_bce['roc_auc']:.6f}")
    print(f"{'Hinge':<14} | {metrics_hinge['accuracy']:.6f} | {metrics_hinge['precision']:.6f}  | {metrics_hinge['recall']:.6f} | {metrics_hinge['f1_score']:.6f} | {metrics_hinge['roc_auc']:.6f}")
    print("-" * 80)
    
    print()
    print("Таблица 6: Влияние L2-регуляризации")
    print("-" * 80)
    print("Параметр lambda | Норма W  | Accuracy | Precision | Recall  | F1-Score | ROC AUC")
    print("-" * 80)
    for lam, m in results_l2.items():
        print(f"{lam:<15} | {m['weight_norm']:.6f} | {m['accuracy']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1_score']:.6f} | {m['roc_auc']:.6f}")
    print("-" * 80)

if __name__ == "__main__":
    main()