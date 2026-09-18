import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def prepare_data(df):
    data = df.copy()
    data.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in data.columns]
    rename = {
        "parental_level_of_education": "parental_education",
        "test_preparation_course": "test_preparation",
        "race_ethnicity": "race_ethnicity",
    }
    data = data.rename(columns=rename)

    score_cols = ["math_score", "reading_score", "writing_score"]
    data["average_score"] = data[score_cols].mean(axis=1)
    # Three interpretable performance classes.
    data["performance_category"] = pd.cut(
        data["average_score"],
        bins=[-1, 39, 59, 100],
        labels=["Low", "Medium", "High"]
    )
    return data

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
