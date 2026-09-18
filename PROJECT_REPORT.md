# PROJECT REPORT

## Student Performance Prediction and Analysis Using Machine Learning


---

## 1. Cover Page

**Project Title:** Student Performance Prediction and Analysis Using Machine Learning

**Student:** Alok Tripathi 
**Registration no:** 25MIM10137
**Program:** Integrated MTech AI
**Institution:** VIT Bhopal  

---

## 2. Introduction

Student performance is influenced by several academic and contextual factors. Educational datasets can be analyzed to discover patterns and build machine-learning applications. This project develops a complete machine-learning workflow using the uploaded StudentsPerformance dataset.

The system loads student records, performs preprocessing, derives a broad performance category from the available subject scores, trains a Random Forest classifier using non-score contextual attributes, evaluates the classifier, and provides an interactive dashboard.

The project demonstrates practical concepts of data preprocessing, feature engineering, supervised learning, evaluation, visualization, modular programming, and testing.

---

## 3. Problem Statement

Raw student records contain useful information but may be difficult to interpret directly. A simple analytical system is required to transform these records into meaningful summaries and provide a reproducible machine-learning prediction workflow.

The proposed system addresses this requirement by creating a modular application for student-performance analysis and broad performance-category prediction.

---

## 4. Objectives

1. Load and validate the student-performance dataset.
2. Clean and transform the data.
3. Create a derived average score and performance category for analytical purposes.
4. Explore subject-wise and category-wise patterns.
5. Build a Random Forest classification model.
6. Evaluate model performance using standard metrics.
7. Develop an interactive Streamlit dashboard.
8. Organize the project into reusable modules.
9. Provide testing and documentation suitable for academic submission.

---

## 5. Dataset Description

The uploaded dataset contains **1000 records and 8 original columns**.

Original columns:
 - gender
 - race/ethnicity
 - parental level of education
 - lunch
 - test preparation course
 - math score
 - reading score
 - writing score

The three score columns are:
- Math score
- Reading score
- Writing score

Other fields describe student/context attributes.

### Derived Features

`average_score` is calculated as:

Average Score = (Math + Reading + Writing) / 3

A three-level analytical target is then created:
- Low: average score 0–39
- Medium: average score 40–59
- High: average score 60–100

The three raw subject scores are excluded from model input so that the model does not directly receive the values used to construct the target.

---

## 6. Functional Requirements

### FR1 — Data Input
The system shall load the CSV dataset.

### FR2 — Data Preprocessing
The system shall normalize column names, calculate derived fields, and prepare categorical and numerical variables.

### FR3 — Exploratory Analysis
The system shall display descriptive statistics and performance distributions.

### FR4 — Model Training
The system shall train a Random Forest classifier.

### FR5 — Prediction
The system shall accept student contextual attributes and output a predicted performance category.

### FR6 — Evaluation
The system shall calculate accuracy, precision, recall, F1-score, confusion matrix, and classification report.

### FR7 — Visualization
The system shall generate charts for subject averages and performance categories.

### FR8 — Dashboard
The system shall provide an interactive Streamlit interface.

---

## 7. Non-Functional Requirements

### Performance
The system should process the supplied dataset without unnecessary computation.

### Usability
The dashboard should present metrics and prediction controls in a simple interface.

### Reliability
The model and preprocessing steps should be reproducible using a fixed random state.

### Maintainability
Functions are separated into modules for preprocessing, feature engineering, training, evaluation, prediction, and visualization.

### Error Handling
The dashboard checks whether the trained model exists before attempting prediction.

### Scalability
The modular pipeline can be extended to larger datasets and additional models.

---

## 8. System Architecture

```text
+-----------------------+
| StudentsPerformance   |
|       CSV Dataset     |
+-----------+-----------+
            |
            v
+-----------------------+
| Data Loading          |
| & Validation          |
+-----------+-----------+
            |
            v
+-----------------------+
| Data Preprocessing    |
| Cleaning + Derived    |
| Features              |
+-----------+-----------+
            |
            +------------------+
            |                  |
            v                  v
+-------------------+   +-------------------+
| Exploratory       |   | Feature           |
| Analysis          |   | Engineering       |
+-------------------+   +---------+---------+
                                  |
                                  v
                         +-------------------+
                         | Random Forest     |
                         | Classifier        |
                         +---------+---------+
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
             +---------------+         +---------------+
             | Evaluation    |         | Prediction    |
             +---------------+         +-------+-------+
                                             |
                                             v
                                     +---------------+
                                     | Streamlit     |
                                     | Dashboard     |
                                     +---------------+
```

---

## 9. Workflow

```text
START
  |
  v
Load CSV
  |
  v
Validate Dataset
  |
  v
Normalize Columns
  |
  v
Calculate Average Score
  |
  v
Create Performance Category
  |
  v
Separate Features and Target
  |
  v
Train/Test Split
  |
  v
Encode Categorical Features
  |
  v
Scale Numerical Features
  |
  v
Train Random Forest
  |
  v
Evaluate Model
  |
  v
Save Model
  |
  v
Launch Dashboard
  |
  v
Predict Student Category
  |
  v
END
```

---

## 10. Use Case Diagram

```text
                 +----------------------------+
                 | Student Performance System |
                 +----------------------------+
                    ^       ^       ^
                    |       |       |
                 +-----+ +-----+ +-------+
                 |User | |User | |Evaluator|
                 +-----+ +-----+ +-------+
                    |       |       |
                    v       v       v
              View Data  Predict   Evaluate
                 |          |         |
                 +----------+---------+
                            |
                            v
                    Analyze Results
```

### Main Use Cases
- View dataset
- Analyze scores
- View performance distribution
- Enter student attributes
- Predict category
- Review evaluation metrics

---

## 11. Sequence Diagram

```text
User -> Dashboard: Open application
Dashboard -> Dataset: Load records
Dataset -> Dashboard: Return data
Dashboard -> User: Display statistics

User -> Dashboard: Enter student attributes
Dashboard -> Model: Send input
Model -> Preprocessor: Transform input
Preprocessor -> Model: Return transformed features
Model -> Dashboard: Return prediction
Dashboard -> User: Display category
```

---

## 12. Component/Class Design

```text
+-----------------------+
| DataPreprocessing     |
+-----------------------+
| load_data()           |
| prepare_data()        |
| split_data()          |
+-----------+-----------+
            |
            v
+-----------------------+
| FeatureEngineering    |
+-----------------------+
| build_preprocessor()  |
+-----------+-----------+
            |
            v
+-----------------------+
| ModelTraining         |
+-----------------------+
| train()               |
+-----------+-----------+
            |
            v
+-----------------------+
| ModelEvaluation       |
+-----------------------+
| evaluate()            |
+-----------+-----------+
            |
            v
+-----------------------+
| Prediction            |
+-----------------------+
| predict_student()     |
+-----------------------+
```

---

## 13. Design Decisions and Rationale

### Random Forest
Random Forest was selected as the primary classifier because it can model nonlinear relationships and works naturally with transformed categorical and numerical features.

### One-Hot Encoding
Categorical values such as gender, parental education, lunch, and test preparation are converted into numerical indicator variables.

### Standard Scaling
Numerical features are standardized within the preprocessing pipeline.

### Pipeline
A Scikit-learn pipeline keeps preprocessing and model inference together.

### Leakage Prevention
Subject scores are not used as model inputs because the performance category is derived from those scores. This prevents the model from simply learning the target from the same variables used to construct it.

---

## 14. Implementation Details

The project is divided into multiple modules:

### data_preprocessing.py
Loads the CSV, normalizes column names, calculates average scores, creates the performance category, and performs train/test splitting.

### feature_engineering.py
Creates the ColumnTransformer used for categorical encoding and numerical scaling.

### model_training.py
Builds and trains the Random Forest pipeline and saves the trained model.

### model_evaluation.py
Loads the trained model and calculates classification metrics.

### exploratory_analysis.py
Generates subject-average and performance-distribution charts.

### prediction.py
Provides a reusable prediction function.

### visualization.py
Contains reusable chart and summary helpers.

### app.py
Implements the Streamlit dashboard.

### test_model.py
Validates dataset loading and preprocessing behavior.

---

## 15. Model Training

The dataset is divided into training and testing subsets using an 80:20 split with stratification and a fixed random state of 42.

The model uses:

**RandomForestClassifier**
- 200 estimators
- random_state = 42
- class_weight = balanced

The trained pipeline is stored as:

`models/student_performance_model.pkl`

---

## 16. Evaluation Methodology

The project evaluates the classifier using:

### Accuracy
Proportion of correct predictions.

### Precision
Measures the proportion of predicted positive/class instances that are correct.

### Recall
Measures how many instances belonging to a class are correctly identified.

### F1-score
Harmonic mean of precision and recall.

### Confusion Matrix
Shows the relationship between actual and predicted classes.

To reproduce the exact current metrics, execute:

```bash
python src/model_evaluation.py
```

---

## 17. Results and Screenshots

The `results/` directory contains generated charts after running:

```bash
python src/exploratory_analysis.py
```

The Streamlit dashboard displays:
- Total students
- Average subject scores
- Dataset preview
- Performance distribution
- Prediction controls
- Subject statistics

Screenshots can be added to this section after running the application.

---

## 18. Testing Approach

Testing includes:
1. Dataset loading test.
2. Non-empty dataset validation.
3. Preprocessing target creation test.
4. Dashboard/model existence validation.
5. Reproducible train/test split.

Run:

```bash
pytest
```

---

## 19. Challenges Faced

### Challenge 1 — Categorical Data
Many input fields are categorical.

**Solution:** One-hot encoding.

### Challenge 2 — Target Leakage
Using subject scores to predict a category created directly from those scores would leak target information.

**Solution:** Exclude the score columns from prediction inputs.

### Challenge 3 — Reproducibility
ML results can vary with random splits.

**Solution:** Fixed random state.

### Challenge 4 — Usability
A command-line model is difficult to demonstrate.

**Solution:** Streamlit dashboard.

---

## 20. Learnings and Key Takeaways

- Learned the complete ML workflow from raw CSV to prediction.
- Learned categorical feature encoding.
- Learned train/test splitting and classification evaluation.
- Learned how to prevent target leakage.
- Learned modular Python project organization.
- Learned basic dashboard development with Streamlit.
- Learned how testing improves project reliability.

---

## 21. Future Enhancements

1. Compare multiple classification algorithms.
2. Add explainable-AI visualizations.
3. Add user authentication.
4. Store prediction history in a database.
5. Deploy the dashboard.
6. Add automated model retraining.
7. Add larger and newer institutional datasets.
8. Add model monitoring.

---

## 22. Conclusion

The Student Performance Prediction and Analysis project demonstrates how machine learning can be applied to a structured educational dataset. The system integrates preprocessing, feature engineering, exploratory analysis, Random Forest classification, evaluation, testing, and an interactive dashboard.

The project provides a complete academic implementation that can be executed locally and extended with additional models, data, and deployment features.

---

## 23. References

1. Scikit-learn documentation — machine learning algorithms and preprocessing.
2. Pandas documentation — data manipulation.
3. Matplotlib documentation — visualization.
4. Streamlit documentation — interactive Python applications.
5. VITyarthi Build Your Own Project submission guidelines.
6. Uploaded StudentsPerformance.csv dataset.
