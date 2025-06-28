# ML & Data Science Projects

This repository contains a collection of projects and assignments completed during a Machine Learning summer school at Oxford in summer of 2024.
Each project explores fundamental ML concepts, data preprocessing, model evaluation, and practical applications using Python libraries such as `numpy`, `pandas`, `scikit-learn`.

---

## Projects

### 1. Walmart Weekly Sales Prediction
**Goal**: Predict weekly sales of Walmart stores using linear regression and polynomial basis functions.

- Preprocessed historical sales data, including holidays and economic indicators.
- Evaluated linear regression vs. polynomial models using MAE & RMSE.
- Explored the effect of dropping certain features on model performance.
- Notebook: `walmart_sales_prediction.ipynb`

### 2. Titanic Survival Prediction
**Goal**: Train a logistic regression model from scratch to predict passenger survival using the Titanic dataset.

- Handled missing data in the `'Age'` column using imputation strategies (median and max value), improving accuracy from ~74% to ~82%.
- Applied one-hot encoding to categorical features (`Sex`, `Embarked`) and standardized numerical inputs.
- Implemented custom logistic regression with gradient descent, sigmoid, and cross-entropy loss from scratch.
- Evaluated the model using accuracy and confusion matrix — final accuracy on test set: **~81.6%**.
- Analysis of model weights revealed **gender** as the most influential factor in survival prediction.
- Notebook: `titanic_survival_prediction.ipynb`
