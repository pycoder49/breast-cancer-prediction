import matplotlib.pyplot as plt
import pandas as pd
import torch

"""
Steps taken:

1) Clean the data -- remove any rows with null values/any columns that are fully null
2) Separate the target and the features
3) Standardize the features -- done manually instead of sklearn library
4) Split the data into training and eval sets
5) Implement logistic regression
6) Evaluate the results
"""


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
        self.b = torch.tensor(0.0)                                  # scalar tensor

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
        self.y_pred = self.forward(self.x_eval_tensor)        # shape: [num_samples, 1] --> [170, 1]

        # applying the threshold
        predictions = torch.where(self.y_pred > 0.5, 1, 0)
        return predictions

    def evaluate(self):
        predictions = self.predict()    # shape: [170, 1]
        accuracy = (predictions == self.y_eval_tensor).float().mean()
        return accuracy


data_df = pd.read_csv("data.csv")


"""
Data cleaning process
"""
# first check if any columns are entirely null
# then drop any rows with null values
null_columns = data_df.isnull().all()  # column "Unnamed: 32" is entirely null
data_df = data_df.drop(columns=["Unnamed: 32"])

# we also drop "id" column since it provides no use
data_df = data_df.drop(columns=["id"])

# diagnosis is out output, which should be 1s and 0s
data_df.diagnosis = [1 if value == "M" else 0 for value in data_df.diagnosis]


"""
Plotting the data
"""
# 357: 0, 212: 1
diagnosis_count = data_df["diagnosis"].value_counts()
plt.bar(diagnosis_count.index, diagnosis_count.values)
plt.xlabel("Diagnosis")
plt.ylabel("Count")
# plt.show()


"""
Separate the target and features
"""
y = data_df.diagnosis
X = data_df.drop(columns=["diagnosis"])


"""
Standardizing the data
"""
for column in X.columns:
    mean = X[column].mean()
    stdev = X[column].std()
    X[column] = (X[column] - mean) / stdev


"""
Splitting the standardized data into test and eval sets
"""
combined_df = pd.concat([y, X], axis=1)
shuffled_combined = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(len(shuffled_combined) * 0.3)  # gives us 30% of the data set

train_dataset = shuffled_combined[split_index:]  # last 70% for training
eval_dataset = shuffled_combined[:split_index]  # first 30% for testing/evaluation

y_train = train_dataset.diagnosis                       # 399 rows                  [399 x 1] matrix
x_train = train_dataset.drop(columns=["diagnosis"])     # 399 rows, 30 columns      [399 x 30] matrix

y_eval = eval_dataset.diagnosis                         # 170 rows                  [170 x 1] matrix
x_eval = eval_dataset.drop(columns=["diagnosis"])       # 170 rows, 30 columns      [170 x 30] matrix


"""
Training the logistic regression model
"""
lr = LogisticRegression(
    x_train=x_train,
    y_train=y_train,
    x_eval=x_eval,
    y_eval=y_eval
)

num_epochs = 1000

for epoch in range(num_epochs):
    lr.update()
    if (epoch + 1) % 100 == 0:
        loss = lr.compute_loss()
        print(f"Loss after epoch {epoch + 1}: {loss.item():.4f}")
print("Training done")
print("-------------")
print(f"Accuracy of the model: {(lr.evaluate().item() * 100):.2f}%")
