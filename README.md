# heart-failure-prediction

## Overview
This project aims to predict heart failure outcomes using a clinical dataset. Various machine learning models are trained and evaluated to determine the best performer. The project involves data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation.

## Dataset
The dataset used in this project is sourced from the [Heart Failure Clinical Records Dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). It contains 299 records of patients with 13 features, including age, anemia, creatinine phosphokinase, diabetes, ejection fraction, and more.

## Installation
To run this project, you need to have Python installed along with the following packages:
```bash
pip install pandas numpy matplotlib seaborn imbalanced-learn scikit-learn xgboost lightgbm

## Usage
Load the dataset:
data = pd.read_csv('path_to_dataset/heart_failure_clinical_records_dataset.csv')

## Data Preprocessing:
Handling missing values
Converting data types
Feature scaling

## Model Training:
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

## Model Evaluation:
Confusion matrix
ROC AUC
Precision-Recall Curve
Models Used
XGBoost Classifier
LightGBM Classifier
Support Vector Machine (SVM)
Gradient Boosting Classifier
Random Forest Classifier
Logistic Regression

## Results
The performance of each model is evaluated based on accuracy, ROC AUC score, and other relevant metrics. Detailed results and comparison are provided in the notebook.

## Future Work
Further feature engineering and selection
Exploring other models and ensemble techniques
Model deployment

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.
