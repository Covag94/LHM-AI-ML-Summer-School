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
- Notebook: `Linear_Regression_HW1_vagianos_final.ipynb`

### 2. Titanic Survival Prediction
**Goal**: Train a logistic regression model from scratch to predict passenger survival using the Titanic dataset.

- Handled missing data in the `'Age'` column using imputation strategies (median and max value), improving accuracy from ~74% to ~82%.
- Applied one-hot encoding to categorical features (`Sex`, `Embarked`) and standardized numerical inputs.
- Implemented custom logistic regression with gradient descent, sigmoid, and cross-entropy loss from scratch.
- Evaluated the model using accuracy and confusion matrix — final accuracy on test set: **~81.6%**.
- Analysis of model weights revealed **gender** as the most influential factor in survival prediction.
- Notebook: `Logistic_Regression_HW2_vagianos_final.ipynb`

### 3. EMNIST Handwritten Character Classification  
**Goal**: Build and compare neural network models to classify handwritten digits and letters from the EMNIST dataset.

- Implemented a manual two-layer neural network from scratch using NumPy (~71% accuracy).  
- Developed fully connected and convolutional neural networks with PyTorch, achieving up to ~86% accuracy.  
- Explored effects of network depth, hidden layer size, and convolutional layers on performance.  
- Included visualizations of training progress and model evaluation metrics.  
- Notebook: `Classification_NN_EMNIST_HW3_vagianos_final.ipynb`

### 4. California Housing Price Regression  
**Goal**: Predict median house prices in California using linear regression with polynomial basis functions.

- Preprocessed the California housing dataset, including feature scaling and polynomial feature expansion.  
- Compared simple linear regression with polynomial regression to capture nonlinear relationships.  
- Evaluated models using Mean Squared Error (MSE) and R² score on training and test sets.  
- Visualized learning curves to analyze underfitting and overfitting behaviors.  
- Explored feature engineering techniques to improve prediction accuracy.  
- Notebook: `Regression_NN_California_HW4_vagianos_final.ipynb`

  ### 5. MVSA Multi-View Sentiment Analysis  
**Goal**: Perform sentiment classification using both images and text from tweets in the MVSA dataset.

- Preprocessed multi-modal data including images and corresponding tweet texts.  
- Implemented three models for sentiment analysis:  
  - CNN-based image-only model  
  - RNN/LSTM-based text-only model  
  - Combined model leveraging both visual and textual inputs.  
- Conducted an 80:20 train-test split ensuring data alignment across modalities.  
- Evaluated models using accuracy and confusion matrices, providing comparative performance analysis.  
- Visualized dataset characteristics including label distribution and image size histograms.  
- Notebook: `Multi_Modal_sentiment_analysis_HW5_final_vagianos.ipynb`
