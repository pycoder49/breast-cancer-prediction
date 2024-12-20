import torch
import pandas as pd


class LogisticRegression:
    """
    Implementing Logistic Regression
    """
    def __init__(self,
                 x_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 x_eval: pd.DataFrame,
                 y_eval: pd.DataFrame,
                 learning_rate=0.01):

        # initialize tensors
        self.x_train_tensor = torch.from_numpy(x_train.values).float()          # shape: [num_samples, num_features]
        self.y_train_tensor = torch.from_numpy(y_train.values).float().reshape(-1, 1)   # shape: [num_samples, 1]
        self.x_eval_tensor = torch.from_numpy(x_eval.values).float()
        self.y_eval_tensor = torch.from_numpy(y_eval.values).float().reshape(-1, 1)
        self.y_pred = torch.zeros_like(self.y_train_tensor)

        # initialize learning rate, weights, and bias
        self.learning_rate = learning_rate
        self.w = torch.zeros(self.x_train_tensor.shape[1], 1)       # shape: [num_features, 1]
        self.b = torch.zeros(1, 1)                                  # scalar tensor

    def forward(self, x):
        # calculating the logit and the sigmoid function --> shape: [num_features, 1]
        z = torch.matmul(x, self.w) + self.b
        sigmoid = 1 / (1 + torch.exp(-z))
        return sigmoid

    def update(self):
        # getting y predictions and calculating dw and db
        self.y_pred = self.forward(self.x_train_tensor)
        dw = (1/self.x_train_tensor.shape[0]) * torch.matmul(self.x_train_tensor.T, (self.y_pred - self.y_train_tensor))
        db = (1/self.x_train_tensor.shape[0]) * torch.sum(self.y_pred - self.y_train_tensor)

        # updating the weights and bias
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def compute_loss(self):
        # binary cross entropy loss equation
        # we don't use matmul since these are element-wise multiplication and not matrix multiplication
        # we also don't use torch.sum because that sums up the entire loss, instead we want the average
        # we also clamp to avoid log(0 / 1-1) and avoid changing the global variable
        y_pred_clamped = torch.clamp(self.y_pred, min=1e-7, max=1 - 1e-7)

        first_half = self.y_train_tensor * torch.log(y_pred_clamped)
        second_half = (1 - self.y_train_tensor) * torch.log(1 - y_pred_clamped)
        bce_loss = -torch.mean(first_half + second_half)
        return bce_loss

    def predict(self):
        # uses the forward function to get a prediction
        y_pred = self.forward(self.x_eval_tensor)        # shape: [num_samples, 1] --> [170, 1]

        # applying the threshold
        predictions = torch.where(y_pred > 0.5, 1, 0)
        return predictions

    def evaluate(self):
        predictions = self.predict()    # shape: [170, 1]
        accuracy = (predictions == self.y_eval_tensor).float().mean()
        return accuracy
