import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, prepare_data, split_data
from feature_engineering import build_preprocessor

DATA_PATH = "dataset/StudentsPerformance.csv"
MODEL_PATH = "models/student_performance_model.pkl"

def train():
    df = prepare_data(load_data(DATA_PATH))

    # Avoid target leakage: raw subject scores are not used as input features.
    X = df.drop(columns=["average_score", "performance_category"])
    y = df["performance_category"]

    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor, _, _ = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model, X_test, y_test

if __name__ == "__main__":
    train()
    print(f"Model saved to {MODEL_PATH}")
