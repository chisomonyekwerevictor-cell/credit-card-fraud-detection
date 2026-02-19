This project focuses on detecting fraudulent transactions in a highly imbalanced dataset of credit card transactions. Fraudulent transactions account for a very small fraction of the total dataset, posing a significant challenge for model training. The goal is to build machine learning models that can effectively identify fraud without overwhelming the system with false positives.

Key Highlights:
	Dataset: A highly imbalanced dataset containing legitimate and fraudulent transactions, with fraudulent transactions making up only 0.17% of the data.
	Project Structure

1. Dataset

The dataset consists of transactions labeled as either legitimate (class 0) or fraudulent (class 1). The data is heavily imbalanced, with the majority class being legitimate transactions. The project uses this data to train models and predict fraudulent transactions.
	•	Features: The dataset includes anonymized features such as v1, v2, v3, etc., representing various transaction characteristics.
	•	Target: The target variable is class, where:
	•	0 indicates legitimate transactions.
	•	1 indicates fraudulent transactions.

2. Data Preprocessing

Class Imbalance Handling
To address the class imbalance, the following techniques were applied:
	•	SMOTE (Synthetic Minority Over-sampling Technique): This technique is used to generate synthetic samples for the minority class (fraudulent transactions) to balance the dataset.
	•	Random Undersampling: The majority class (legitimate transactions) is randomly undersampled to a smaller size, ensuring a balanced dataset for model training.

Feature Scaling
	•	StandardScaler was applied to normalize the features. This ensures that the model doesn’t give undue importance to features with larger scales, which is especially important for Logistic Regression.

3. Model Training

Logistic Regression
	•	Solver: saga solver is used, suitable for large datasets with L1 regularization.
	•	Class Weights: class_weight='balanced' is used to adjust for the class imbalance.
	•	Max Iterations: The model is trained with max_iter=5000 to ensure convergence.

Random Forest
	•	Class Weights: The class_weight='balanced' parameter ensures that the model places more emphasis on the minority class (fraudulent transactions).
	•	Hyperparameters:
	•	n_estimators=300: The number of trees in the forest.
	•	max_depth=10: Controls the depth of each tree to prevent overfitting.
	•	min_samples_leaf=4: Ensures each leaf node contains at least 4 samples.

XGBoost
	•	Class Weights: scale_pos_weight is adjusted according to the imbalance ratio (583.2:1).
	•	Hyperparameters:
	•	n_estimators=300: Number of boosting rounds (trees).
	•	learning_rate=0.05: Controls the contribution of each tree.
	•	max_depth=6: Controls the depth of each tree to prevent overfitting.
	•	subsample=0.8: Specifies the fraction of samples to use for each tree, adding randomness to reduce overfitting.
4. Threshold Tuning
Since fraud detection is sensitive to false positives and false negatives, thresholds for the models were adjusted to optimize performance:
	•	Logistic Regression: Threshold of 0.05 (model is highly conservative in predicting fraud).
	•	Random Forest: Threshold of 0.5 (default threshold for binary classification).
	•	XGBoost: Threshold of 0.8 (model is conservative in predicting fraud).

This helps optimize for precision and recall, ensuring that fraud is detected while minimizing the number of false positives.
Model Evaluation
Models are evaluated using the following metrics:
1. Precision: Measures how many of the predicted fraudulent transactions are actually fraudulent.

2. Recall: Measures how many of the actual fraudulent transactions are correctly identified.

3. F1-Score: The harmonic mean of precision and recall, used to balance both metrics.

4. ROC-AUC: Measures the ability of the model to distinguish between the positive (fraudulent) and negative (legitimate) classes.
	•	Random Forest and XGBoost achieved perfect recall (1.00) and F1-score (1.00) for fraudulent transactions.
	•	Logistic Regression performed well in terms of recall (1.00), but precision for fraudulent transactions was low (0.01), leading to a low F1-score for fraud detection.

Example Results:

Model	Precision (Fraud)	Recall (Fraud)	F1-Score (Fraud)	Accuracy	ROC-AUC
Logistic Regression	0.01	1.00	0.02	0.51	0.999
Random Forest	0.67	1.00	1.00	1.00	0.9999
XGBoost	0.26	1.00	0.41	0.99	0.9999
Installation
Requirements:
	•	pandas
	•	numpy
	•	scikit-learn
	•	xgboost
	•	imbalanced-learn
	•	matplotlib
	•	seaborn
	•	jupyter

Usage
	1.	Data Preprocessing: Start by running the preprocessing steps in the Jupyter notebook (fraud_detection_notebook.ipynb).
	2.	Model Training: The notebook guides you through training Logistic Regression, Random Forest, and XGBoost models.
	3.	Threshold Tuning: Adjust thresholds for each model to find the optimal trade-off between precision and recall.
	4.	Evaluation: Evaluate the performance of each model based on precision, recall, F1-score, and ROC-AUC.
Models Used: Logistic Regression, Random Forest, and XGBoost.
	Class Imbalance Handling: Techniques like SMOTE (Synthetic Minority Over-sampling Technique) and undersampling were employed to address class imbalance.
	Evaluation Metrics: The performance of models is evaluated using precision, recall, F1-score, and ROC-AUC. F1-score is prioritized as it strikes a balance between precision and recall, essential for fraud detection tasks.
