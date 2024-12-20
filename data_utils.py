import pandas as pd
import matplotlib.pyplot as plt


def clean_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning process

    Args:
        :param data_df: Takes a dataframe of a csv file

    :return data_df: Cleaned data with no null values and modified output values
    """
    # first check if any columns are entirely null
    # then drop any rows with null values
    null_columns = data_df.isnull().all()  # column "Unnamed: 32" is entirely null
    data_df = data_df.drop(columns=["Unnamed: 32"])

    # we also drop "id" column since it provides no use
    data_df = data_df.drop(columns=["id"])

    # diagnosis is out output, which should be 1s and 0s
    data_df.diagnosis = [1 if value == "M" else 0 for value in data_df.diagnosis]

    return data_df


def prep_data(data_df: pd.DataFrame, split_ratio=0.3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepares the data for training and evaluation.

    Args:
        :param data_df: Cleaned dataset with features and target
        :param split_ratio: Percentage of the dataset to be kept for training

    :returns:
        tuple: Tuple containing:
            - x_train_set (pd.DataFrame): Training features.
            - y_train_set (pd.DataFrame): Training target labels.
            - x_eval_set (pd.DataFrame): Evaluation features.
            - y_eval_set (pd.DateFrame): Evaluation target labels.
    """
    # separating targets and features
    y = data_df.diagnosis
    x = data_df.drop(columns=["diagnosis"])

    # standardizing the data using pandas built-in methods
    x = (x - x.mean()) / x.std()

    # splitting standardized data into training and testing/eval sets
    combined_df = pd.concat([y, x], axis=1)
    shuffled_combined = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int(len(shuffled_combined) * split_ratio)  # gives us 30% of the data set

    train_dataset = shuffled_combined[split_index:]  # last 70% for training
    eval_dataset = shuffled_combined[:split_index]  # first 30% for testing/evaluation

    y_train_set = train_dataset.diagnosis                       # 399 rows                  [399 x 1] matrix
    x_train_set = train_dataset.drop(columns=["diagnosis"])     # 399 rows, 30 columns      [399 x 30] matrix

    y_eval_set = eval_dataset.diagnosis                         # 170 rows                  [170 x 1] matrix
    x_eval_set = eval_dataset.drop(columns=["diagnosis"])       # 170 rows, 30 columns      [170 x 30] matrix

    return x_train_set, y_train_set, x_eval_set, y_eval_set


def plot_data(data_df: pd.DataFrame):
    # 357: 0, 212: 1
    diagnosis_count = data_df["diagnosis"].value_counts()
    plt.bar(diagnosis_count.index, diagnosis_count.values)      # bar graph
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.show()
