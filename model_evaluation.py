import os
import sys
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, prepare_data, split_data

MODEL_PATH = "models/student_performance_model.pkl"
DATA_PATH = "dataset/StudentsPerformance.csv"

def evaluate():
    df = prepare_data(load_data(DATA_PATH))
    X = df.drop(columns=["average_score", "performance_category"])
    y = df["performance_category"]

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    }
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    evaluate()
