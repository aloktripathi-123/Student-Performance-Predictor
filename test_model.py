import os
import sys
sys.path.append("src")
from data_preprocessing import load_data, prepare_data

def test_dataset_loads():
    df = load_data("dataset/StudentsPerformance.csv")
    assert len(df) > 0
    assert df.shape[1] >= 5

def test_preprocessing_creates_target():
    df = prepare_data(load_data("dataset/StudentsPerformance.csv"))
    assert "average_score" in df.columns
    assert "performance_category" in df.columns
    assert df["performance_category"].notna().all()
