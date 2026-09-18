import pandas as pd
import matplotlib.pyplot as plt

def subject_summary(df):
    return df[["math_score", "reading_score", "writing_score"]].mean()

def performance_summary(df):
    return df["performance_category"].value_counts()

def save_chart(series, title, ylabel, path):
    plt.figure(figsize=(7,5))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
