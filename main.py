import pandas as pd
import data_utils as utils
import models

"""
Steps taken:

1) Clean the data -- remove any rows with null values/any columns that are fully null
2) Separate the target and the features
3) Standardize the features -- done manually instead of sklearn library
4) Split the data into training and eval sets
5) Implement logistic regression

**Code was organized into different files after**
"""

data = pd.read_csv("data.csv")

data = utils.clean_data(data)
# plot_data(data)

x_train, y_train, x_eval, y_eval = utils.prep_data(data)

# training the logistic regression model
lr = models.LogisticRegression(
    x_train=x_train,
    y_train=y_train,
    x_eval=x_eval,
    y_eval=y_eval
)

# training loop
num_epochs = 1000

for epoch in range(num_epochs):
    lr.update()

    if (epoch + 1) % 100 == 0:
        loss = lr.compute_loss()
        print(f"Loss after epoch {epoch + 1}: {loss.item():.4f}")

accuracy = lr.evaluate().item() * 100

print("-------------")
print("Training complete")
print(f"Accuracy of the model: {accuracy:.2f}%")
