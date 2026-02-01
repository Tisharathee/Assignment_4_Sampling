# Sampling Techniques for Imbalanced Credit Card Dataset

Name: Tanishka

Roll No: 102303245

Assignment Title: Sampling Techniques on Imbalanced Dataset

## Objective

The main objective of this assignment is to understand the importance of sampling techniques in handling imbalanced datasets and to analyze how different sampling strategies influence the 

performance of various machine learning models.

Imbalanced datasets are common in real-world problems such as fraud detection, spam detection, and medical diagnosis, where one class appears much less frequently than the other.

## Dataset Description

The dataset used is a Credit Card Fraud Detection dataset.

Source: Provided via GitHub link

Total Features: 31

Target Column: Class

0 → Legitimate transaction

1 → Fraudulent transaction

Initially, the dataset is highly imbalanced with very few fraud cases compared to legitimate transactions.

## Problem with Imbalanced Data

In an imbalanced dataset:

Machine learning models become biased towards the majority class.

Accuracy becomes misleading.
Sampling Techniques Used

The following five sampling techniques were applied:

Minority class (important class) is poorly predicted.

To overcome this issue, different sampling techniques are applied to balance the dataset before training models.

| Sampling Technique   | Description                          |
| -------------------- | ------------------------------------ |
| Random UnderSampling | Reduces majority class samples       |
| Random OverSampling  | Duplicates minority class samples    |
| SMOTE                | Generates synthetic minority samples |
| NearMiss             | Selects closest majority samples     |
| SMOTE + Tomek        | Removes overlapping noisy samples    |

## Machine Learning Models Used

Five machine learning models were trained on each sampled dataset:

| Model ID | Algorithm           |
| -------- | ------------------- |
| M1       | Logistic Regression |
| M2       | Decision Tree       |
| M3       | Random Forest       |
| M4       | K-Nearest Neighbors |
| M5       | Naive Bayes         |

## Accuracy Results (Actual Output):


| Model               | Random Under | Random Over | SMOTE      | NearMiss   | SMOTE+Tomek |
| ------------------- | ------------ | ----------- | ---------- | ---------- | ----------- |
| Logistic Regression | 92.06        | 93.46       | 93.46      | 93.46      | 89.86       |
| Decision Tree       | 97.66        | 98.60       | 98.60      | 98.13      | 96.14       |
| Random Forest       | **100.00**   | **100.00**  | **100.00** | **100.00** | 99.03       |
| KNN                 | 82.71        | 85.98       | 85.98      | 83.18      | 85.99       |
| Naive Bayes         | 83.18        | 86.45       | 86.45      | 81.78      | 83.09       |


## Best Performing Results

Highest Accuracy Overall

**Random Forest (M3) achieved 100% accuracy with:**

**Random UnderSampling**

Random OverSampling

SMOTE

NearMiss

Best Sampling Techniques (Overall)

Random OverSampling & SMOTE gave consistently high performance across most models.

Worst Performing Technique

NearMiss showed relatively lower accuracy for KNN and Naive Bayes due to aggressive removal of majority samples.

## Conclusion

This experiment demonstrates that sampling techniques significantly improve machine learning performance on imbalanced datasets.

**Key observations:**

Oversampling methods such as Random OverSampling and SMOTE provide the most stable results.

Random Forest is the most robust model for this dataset.

Undersampling methods like NearMiss may lead to information loss.

Hence, selecting an appropriate sampling strategy combined with a strong model is essential for real-world imbalanced classification problems like fraud detection.

## Technologies Used

Python

Pandas, NumPy

Scikit-learn

Imbalanced-learn

Google Colab / Jupyter Notebook
