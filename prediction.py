import joblib
import pandas as pd

MODEL_PATH = "models/student_performance_model.pkl"

def predict_student(student_data):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([student_data])
    return model.predict(X)[0]

if __name__ == "__main__":
    sample = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation": "completed",
    }
    print("Prediction:", predict_student(sample))
