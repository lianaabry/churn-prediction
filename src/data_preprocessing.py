import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """
    Load dataset from csv file
    """
    df = pd.read_csv(path)
    return df

def check_missing_data(df):
    """
    Return missing value summary
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_table = pd.DataFrame({
        "missisng_values": missing,
        "percent": missing_percent
    })
    return missing_table

def clean_data(df):
    """
    Convert columns to appropraite data types
    """
    categorical_columns = ["Geography", "Gender"]

    for col in categorical_columns:
        df[col] = df[col].astype("category")

    return df

def split_features_target(df, target="Exited"):
    """
    Split dataset into X and y
    """
    X = df.drop(target, axis=1)
    y = df[target]

    return X, y

def train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    train_set.to_csv("data/train.csv", index=False)
    test_set.to_csv("data/test.csv", index=False)

    return X_train, X_test, y_train, y_test



