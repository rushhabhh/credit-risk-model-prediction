# Credit Risk Model Prediction

## Overview  
This project focuses on **predicting the creditworthiness of loan applicants** using machine learning. The goal is to identify whether an individual is likely to **default on a loan**, helping financial institutions minimize risk and make data-driven lending decisions.  

By applying advanced ML algorithms, feature engineering, and model evaluation techniques, this project demonstrates how predictive analytics can be applied to **credit risk assessment** in the banking and finance domain.

## Objectives
- Predict whether a loan applicant will **default or repay** the loan.  
- Build robust ML models using **imbalanced classification techniques**.  
- Evaluate performance across multiple algorithms using ROC-AUC, F1-Score, and Precision-Recall metrics.  

## Methodology

### 1. Data Preprocessing
- Handled missing values and outliers  
- Encoded categorical features using one-hot encoding and label encoding  
- Scaled numerical features using StandardScaler  
- Applied **SMOTE** to balance imbalanced classes  

### 2. Exploratory Data Analysis (EDA)
- Visualized correlations between features and target variable  
- Identified key indicators of loan default (income, credit history, etc.)  
- Checked feature distributions and class imbalance  

### 3. Model Building
- Implemented and compared multiple classification models:
  - Logistic Regression  
  - Random Forest Classifier  
  - XGBoost Classifier  
  - Support Vector Machine (SVM)  
- Tuned hyperparameters using GridSearchCV and RandomizedSearchCV  

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score  
- ROC-AUC Curve comparison  
- Confusion Matrix visualization

## Results
| Model | Accuracy | ROC-AUC | F1-Score |
|--------|-----------|----------|-----------|
| Logistic Regression | 0.83 | 0.86 | 0.79 |
| Random Forest | 0.88 | 0.91 | 0.85 |
| XGBoost | **0.90** | **0.93** | **0.88** |

> ✅ **XGBoost achieved the best overall performance**, demonstrating strong predictive power and stability on the test data.


## Key Insights
- Credit history, annual income, and loan amount were the top predictive features.  
- Feature scaling and SMOTE balancing significantly improved recall for the minority (default) class.  
- Ensemble models like XGBoost outperformed linear models in handling nonlinear relationships.

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, streamlit
- **Environment:** Jupyter Notebook

## How to Run
```bash
# Clone this repository
git clone https://github.com/<your-username>/credit-risk-prediction.git

# Navigate to the project folder
cd credit-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook Credit_Risk_Model_Prediction.ipynb

Also play around by the deployed app: https://credit-risk-model-prediction-finance.streamlit.app/
