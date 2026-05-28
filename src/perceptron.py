import numpy as np

class Perceptron:
    def __init__(
        self,
        n_features: int = 2,
        lr: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        init_type: str = "random",
        loss_type: str = "bce",
        lambda_reg: float = 0.0
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_type = init_type
        self.loss_type = loss_type
        self.lambda_reg = lambda_reg

        if init_type == "random":
            self.w = np.random.randn(n_features) * 0.01
        
        elif init_type == "zeros":
            self.w = np.zeros(n_features)
        
        elif init_type == "large":  
            self.w = np.random.normal(0, 10, n_features) 

        self.b = 0.0
        self.loss_history_train = []
        self.loss_history_test = []


    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    
    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]

        y_pred = np.clip(y_pred, 1e-15, 1.0-1e-15)

        loss = (-1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(loss)

    @staticmethod
    def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_hinge = 2 * y_true - 1
        y_pred = np.clip(y_pred, -1 + 1e-15, 1 - 1e-15)

        loss = np.maximum(0, 1 - y_true_hinge * y_pred)
        return float(np.mean(loss))

    def gradient_descent(self, X_batch: np.ndarray, y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> tuple[np.ndarray, float]:
        m = X_batch.shape[0]

        diff = y_pred_batch - y_true_batch

        dw = (X_batch.T @ diff) / m
        db = diff.sum() / m

        return dw, float(db)

    def lin_forward(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return z
    
    def forward(self, X:np.ndarray) -> np.ndarray:
        return self.sigmoid(self.lin_forward(X))
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        n_samples = X_train.shape[0]
        self.loss_history_test = []
        self.loss_history_train = []

        for epoch in range(self.epochs):
            indexes = np.random.permutation(n_samples)
            X_train_shuffle = X_train[indexes]
            y_train_shuffle = y_train[indexes]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_train_shuffle[i: i + self.batch_size]
                y_batch = y_train_shuffle[i: i + self.batch_size]
                m_b = X_batch.shape[0]

                if self.loss_type == "bce":
                    y_pred_batch = self.forward(X_batch)
                    dw, db = self.gradient_descent(X_batch, y_batch, y_pred_batch)
                
                elif self.loss_type == "hinge_loss":
                    z_batch = self.lin_forward(X_batch)
                    y_true_hinge = 2 * y_batch - 1

                    mask = (y_true_hinge * z_batch < 1).astype(float)
                    grad_coeff = -y_true_hinge * mask
                    
                    dw = (X_batch.T @ grad_coeff) / m_b
                    db = np.sum(grad_coeff) / m_b

                else:
                    raise ValueError(f"Unknown type of loss function: {self.loss_type}")

                dw += self.lambda_reg * self.w

                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            if self.loss_type == "bce":    
                y_train_pred_all = self.forward(X_train)
                train_loss = self.compute_loss(y_train, y_train_pred_all)
                y_test_pred_all = self.forward(X_test)
                test_loss = self.compute_loss(y_test, y_test_pred_all)

            elif self.loss_type == "hinge_loss":
                z_train_all = self.lin_forward(X_train)
                train_loss = self.hinge_loss(y_train, z_train_all)
                z_test_all = self.lin_forward(X_test)
                test_loss = self.hinge_loss(y_test, z_test_all)

            else:
                raise ValueError(f"Unknown type of loss function: {self.loss_type}")

            l2_penalty = 0.5 * self.lambda_reg * np.sum(self.w ** 2)
            self.loss_history_train.append(train_loss + l2_penalty)
            self.loss_history_test.append(test_loss + l2_penalty)

    def predict_probability(self, X:np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X:np.ndarray) -> np.ndarray:
        probability = self.predict_probability(X)
        return (probability >= 0.5).astype(int)
