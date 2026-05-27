import numpy as np
import matplotlib.pyplot as plt
import os

from tests import run_experiment
from data_prepare import *
from utilities import *

def main() -> None: 
    os.makedirs("graphics", exist_ok= True)

    X, y = generate_linear_gauss(n_samples=500, noise_level=0.1, random_state=42)
    X_train, y_train, X_test, y_test = split_data(X, y, train_part=0.7, random_state=42)
    X_train_scaled, X_test_scaled, mean, std = stadartizate_data(X_train, X_test)

    results_lr = {}
    results_batch = {}
    results_init = {}

    lr_variants = [0.001, 0.01, 0.5, 1.0]
    batch_variants = [1, 16, 64, 256]
    
    n_features = X_train_scaled.shape[1]

    for idx, current_lr in enumerate(lr_variants):
        results_lr[current_lr] = run_experiment(n_features, X_train_scaled, y_train, X_test_scaled, y_test, current_lr, 32, "random", exp_name=f"lr_{current_lr}")

    for idx, current_batch in enumerate(batch_variants):
        results_batch[current_batch] = run_experiment(n_features, X_train_scaled, y_train, X_test_scaled, y_test, 0.1, current_batch, "random", exp_name= f"batch_{current_batch}")

    results_init["zeros"] = run_experiment(n_features, X_train_scaled, y_train, X_test_scaled, y_test, 0.1, 32, init_type="zeros", exp_name="init_zeros")
    results_init["random"] = run_experiment(n_features, X_train_scaled, y_train, X_test_scaled, y_test, 0.1, 32, init_type="random", exp_name="init_random")
    results_init["large"] = run_experiment(n_features, X_train_scaled, y_train, X_test_scaled, y_test, 0.1, 32, init_type="large", exp_name="init_large")

    print()
    print("Таблица 1: Влияние скорости обучения")
    print("-" * 35)
    print("Параметр lr  | Итоговый Accuracy")
    print("-" * 35)
    for k, v in results_lr.items():
        print(f"{k:<12} | {v}")
    print("-" * 35)

    print("Таблица 2: Влияние размера батча (batch_size)")
    print("-" * 35)
    print("Размер батча  | Итоговый Accuracy")
    print("-" * 35)
    for k, v     in results_batch.items():
        print(f"{k:<13} | {v}")
    print("-" * 35)

    print("Таблица 3: Влияние инициализации весов")
    print("-" * 35)
    print("Тип инициализации  | Итоговый Accuracy")
    print("-" * 35)
    for k, v in results_init.items():
        print(f"{k:<18} | {v}")
    print("-" * 35)


if __name__ == "__main__":
    main()