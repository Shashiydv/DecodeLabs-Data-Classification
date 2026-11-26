# DecodeLabs AI Internship — Project 2: Data Classification Using AI

## Overview
A Supervised Machine Learning classification pipeline built in Python using **Scikit-Learn** to classify Iris flower species (**Setosa**, **Versicolor**, and **Virginica**) based on sepal and petal dimensions.

## Architecture & Workflow (IPO Framework)
- **Input:** Iris Benchmark Dataset (150 balanced samples, 4 numerical features, 3 target classes).
- **Process:**
  - **Train-Test Split:** 80% training and 20% testing split with shuffling to eliminate order bias.
  - **Feature Scaling:** Standardized features (`Mean = 0`, `Variance = 1`) using `StandardScaler`.
  - **Model Training:** Trained a **K-Nearest Neighbors (`KNeighborsClassifier`)** model (`K=5`) with Elbow Method error-rate verification.
- **Output & Validation:** Evaluated model reliability beyond raw accuracy using a **Confusion Matrix**, **Precision**, **Recall**, and **F1 Score**.

## Tech Stack
- Python 3
- Scikit-Learn (`sklearn`)
- Pandas

## How to Run
1. Install dependencies:
   ```bash
   pip install scikit-learn pandas