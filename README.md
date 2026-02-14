# 💳 Credit Card Default Prediction — ML Assignment 2

## a. Problem Statement

Predict whether a credit card client will **default on their payment next month** based on demographic features, credit history, bill statements, and past payment records. This is a **binary classification problem** (Default = 1, No Default = 0).

## b. Dataset Description

- **Dataset:** Default of Credit Card Clients (UCI ML Repository)
- **Instances:** 30,000
- **Features:** 23 (after dropping the ID column)
- **Target Variable:** `target` (originally `default.payment.next.month`)
- **Class Distribution:** ~77.88% No Default, ~22.12% Default (imbalanced)

### Full Feature List (23 features):
| Category | Feature | Description |
|----------|---------|-------------|
| **Demographic** | `LIMIT_BAL` | Amount of given credit (NT dollar) |
| | `SEX` | Gender (1=male, 2=female) |
| | `EDUCATION` | Education (1=grad school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown) |
| | `MARRIAGE` | Marital status (1=married, 2=single, 3=others) |
| | `AGE` | Age in years |
| **History** | `PAY_0` | Repayment status in Sept, 2005 (-1=pay duly, 1=payment delay for one month, 2=two months, ... 9=nine months and above) |
| | `PAY_2` | Repayment status in August, 2005 |
| | `PAY_3` | Repayment status in July, 2005 |
| | `PAY_4` | Repayment status in June, 2005 |
| | `PAY_5` | Repayment status in May, 2005 |
| | `PAY_6` | Repayment status in April, 2005 |
| **Bill Amount**| `BILL_AMT1` | Amount of bill statement in Sept, 2005 (NT dollar) |
| | `BILL_AMT2` | Amount of bill statement in August, 2005 |
| | `BILL_AMT3` | Amount of bill statement in July, 2005 |
| | `BILL_AMT4` | Amount of bill statement in June, 2005 |
| | `BILL_AMT5` | Amount of bill statement in May, 2005 |
| | `BILL_AMT6` | Amount of bill statement in April, 2005 |
| **Payment** | `PAY_AMT1` | Amount of previous payment in Sept, 2005 (NT dollar) |
| | `PAY_AMT2` | Amount of previous payment in August, 2005 |
| | `PAY_AMT3` | Amount of previous payment in July, 2005 |
| | `PAY_AMT4` | Amount of previous payment in June, 2005 |
| | `PAY_AMT5` | Amount of previous payment in May, 2005 |
| | `PAY_AMT6` | Amount of previous payment in April, 2005 |

## c. Models Used

The following 6 classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Naive Bayes (Gaussian) Classifier
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.8163 | 0.7444 | 0.6547 | 0.3587 | 0.4635 | 0.3879 |
| KNN | 0.8003 | 0.7102 | 0.5820 | 0.3451 | 0.4333 | 0.3378 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Random Forest (Ensemble) | 0.8167 | 0.7734 | 0.6615 | 0.3504 | 0.4581 | 0.3865 |
| XGBoost (Ensemble) | 0.8170 | 0.7772 | 0.6592 | 0.3572 | 0.4633 | 0.3895 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves 80.77% accuracy and the highest precision (0.6868) among all models, meaning when it predicts a default, it is most often correct. However, it has the lowest recall (0.2396), indicating it misses ~76% of actual defaulters. AUC of 0.7076 is the lowest, showing limited discriminatory power. Best suited as a quick, interpretable baseline. |
| Decision Tree | Accuracy improves to 81.63% with better recall (0.3587) than Logistic Regression — catches more actual defaulters. AUC jumps to 0.7444. F1 score (0.4635) is significantly better than LR's 0.3553, indicating improved precision-recall balance. With max_depth=6, overfitting is well controlled. |
| KNN | Lowest accuracy (80.03%) among non-NB models. Precision drops to 0.5820 and recall (0.3451) lags behind tree-based models. AUC is 0.7102. With 23 features, the curse of dimensionality limits KNN's distance-based approach. MCC of 0.3378 is the second lowest overall. |
| Naive Bayes (Gaussian) | Lowest accuracy (75.25%) but achieves the highest recall (0.5539) by a large margin — catches over 55% of actual defaulters. Trade-off: precision drops to 0.4515 (many false positives). Despite this, it gets the best F1 score (0.4975). Useful for screening where missing a defaulter is more costly than false alarms. |
| Random Forest (Ensemble) | Second-highest accuracy (81.67%) with the second-best AUC (0.7734). Precision (0.6615) is solid. As a bagging ensemble, it reduces the variance seen in a single Decision Tree, leading to more stable predictions. MCC of 0.3865 shows good overall prediction quality. |
| XGBoost (Ensemble) | Best overall performer — highest accuracy (81.70%), highest AUC (0.7772), and highest MCC (0.3895). As a boosting ensemble, it sequentially corrects errors from weaker learners. Marginally outperforms Random Forest across all key metrics, making it the most reliable model for this credit default prediction task. |

## Project Structure

```
ml_assignment_2/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
├── data/
│   ├── dataset.csv         # Original dataset
│   └── test_data.csv       # Test split for Streamlit app
└── model/
    ├── model_training.ipynb # Model training notebook 
    ├── scaler.pkl           # StandardScaler
    ├── feature_names.pkl    # Feature column names
    ├── model_results.csv    # Comparison table results
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit App Features

- **Dataset Upload (CSV):** Upload test data for evaluation
- **Model Selection Dropdown:** Choose from 6 trained ML models
- **Evaluation Metrics Display:** Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion Matrix & Classification Report:** Visual and tabular results
- **Compare All Models:** Side-by-side comparison table on uploaded data
