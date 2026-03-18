import pandas as pd

def drop_unused_columns(df):
    """
    Remove columns that are not useful for prediction
    """
    cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(cols_to_drop, axis=1)
    return df

def create_age_groups(df):
    """
    Create age category feature
    """
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[18,30,40,50,60,100],
        labels=["Young", "Adult", "MidAge", "Senior", "Old"]
    )
    return df

def create_balance_features(df):
    """
    Create balance related features
    """
    df["HasBalance"] = (df["Balance"] > 0).astype(int)
    df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"]
    return df

def create_product_features(df):
    """
    Features based on product usage
    """
    df["IsSingleProduct"] = (df["NumOfProducts"] ==1).astype(int)
    return df

def create_loyality_feature(df):
    """
    Create loyality indicator
    """
    df["LoyalCustumer"] = (
        (df["Tenure"] > 5) & (df["IsActiveMember"] == 1)
    ).astype(int)
    return df

def encode_categorical(df):
    """
    One-hot encode categorical features
    """
    df = pd.get_dummies(df, columns=["Gender", "Geography", "AgeGroup"], drop_first=True)
    return df

def feature_engineering_pipeline(df):
    """
    Run full feature engineering pipeline
    """
    df = drop_unused_columns(df)

    df = create_age_groups(df)

    df = create_balance_features(df)

    df = create_product_features(df)

    df = create_loyality_feature(df)

    df = encode_categorical(df)

    return df


