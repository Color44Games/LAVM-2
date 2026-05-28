import numpy as np
from sklearn.datasets import make_classification

def generate_data(n_samples: int = 500, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1,
        flip_y= 0.1
    )

    return X, y

def generate_linear_gauss(n_samples: int = 500, centers = [(-2, -2), (2, 2)], cov = [[1.0, 0.0], [0.0, 1.0]], noise_level: float = 0.0, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_state)
    n_per_class = n_samples // 2

    mean0 = np.array(centers[0])
    X0 = np.random.multivariate_normal(mean0, cov, n_per_class)

    mean1 = np.array(centers[1])
    X1 = np.random.multivariate_normal(mean1, cov, n_samples - n_per_class)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_samples - n_per_class)])

    if noise_level > 0:
        noisy_idx = np.random.choice(n_samples, size=int(n_samples * noise_level), replace=False)
        y[noisy_idx] = 1 - y[noisy_idx]

    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]

    return X, y

def generate_xor(n_samples: int = 500, noise_level: float = 0.0, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_state)
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])

    n_per_corner = n_samples // 4
    other = n_samples % 4

    X_list = []
    y_list = []

    for i, corner in enumerate(corners):
        count = n_per_corner + (1 if i < other else 0)
        points = corner + np.random.normal(0, 0.05, (count, 2))
        X_list.append(points)
        y_list.append(np.full(count, labels[i]))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    if noise_level > 0:
        noisy_idx = np.random.choice(n_samples, size=int(n_samples * noise_level), replace=False)
        y[noisy_idx] = 1 - y[noisy_idx]

    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]

    return X, y

def generate_circle(n_samples: int = 500, radius: float = 1.5, noise_level: float = 0.0, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_state)
    
    X = np.random.uniform(-2, 2, (n_samples, 2))
    distance = np.linalg.norm(X, axis=1)
    y = (distance > radius).astype(int)

    if noise_level > 0:
        noisy_idx = np.random.choice(n_samples, size=int(n_samples * noise_level), replace=False)
        y[noisy_idx] = 1 - y[noisy_idx]

    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]

    return X, y

def split_data(X: np.ndarray, y:np.ndarray, train_part: float = 0.7, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(random_state)    

    index_0 = np.where(y == 0)[0]
    index_1 = np.where(y == 1)[0]

    index_0 = np.random.permutation(index_0)
    index_1 = np.random.permutation(index_1)

    train_count_0 = int(len(index_0) * train_part)
    train_count_1 = int(len(index_1) * train_part)

    train_index_0 = index_0[:train_count_0]
    train_index_1 = index_1[:train_count_1]
    test_index_0 = index_0[train_count_0:]
    test_index_1 = index_1[train_count_1:]

    train_index = np.concatenate([train_index_0, train_index_1])
    test_index = np.concatenate([test_index_0, test_index_1])

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    return X_train, y_train, X_test, y_test

def standartizate_data(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std = np.where(std == 0, 1e-10, std) 

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled