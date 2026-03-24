# Customer Churn Prediction

This project builds a Machine Learning pipeline to predict whether a customer will leave a bank (churn) based on their account information and behavior.

The goal is to help businesses identify customers at risk of leaving and take proactive actions to retain them.

---

## Dataset

The project uses a bank customer churn dataset similar to the well-known dataset used in many ML tutorials.

Features include:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

Target variable:

- **Exited**
    - 0 → customer stays
    - 1 → customer churns

---

## Project Structure
churn-prediction/

data/
    churn.csv

src/
    data_preprocessing.py
    feature_engineering.py
    model_training.py
    model_evaluation.py

notebooks/
    eda.ipynb

main.py
requirements.txt
README.md
.gitignore

---

## Machine Learning Pipeline

The project follows a structured ML workflow:

1. Data Loading
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Saving

---

## Models Used

The following models are implemented:

- Logistic Regression
- Random Forest Classifier

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Key Features

Feature engineering includes:

- Age grouping
- Balance indicators
- Loyalty features
- Product usage features
- Balance-to-salary ratio

These features help capture customer behavior patterns.

---

## Plots 

All plots saved in Plots directory

- confusion matrix
- roc curve 
- sham summary
- shap summary bar
- shap summary for the first example

## Example Results

Example evaluation metrics:

Accuracy: 0.86  
ROC-AUC: 0.89  

Random Forest typically performs best on this dataset.

---

## Installation

Clone the repository:

git clone  https://github.com/lianaabry/churn-prediction.git
cd churn-prediction


Install dependencies:

pip install -r requirements.txt


---

## Running the Project

Run the training pipeline:

python main.py


---

## Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- SHAP feature importance
- Model deployment (API)

---


