import os
import matplotlib.pyplot as plt
import pandas as pd
from data_preprocessing import load_data, prepare_data

DATA_PATH = "dataset/StudentsPerformance.csv"
OUT = "results"

def generate():
    os.makedirs(OUT, exist_ok=True)
    df = prepare_data(load_data(DATA_PATH))

    plt.figure(figsize=(7,5))
    df[["math_score","reading_score","writing_score"]].mean().plot(kind="bar")
    plt.ylabel("Average Score")
    plt.title("Average Score by Subject")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"average_subject_scores.png"))
    plt.close()

    plt.figure(figsize=(7,5))
    df["performance_category"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Performance Category")
    plt.ylabel("Number of Students")
    plt.title("Performance Category Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"performance_distribution.png"))
    plt.close()

if __name__ == "__main__":
    generate()
