# A-Comparative-Analysis-of-Machine-Learning-Models-for-Heart-Disease-Diagnosis-Prediction
This project compares four machine-learning models (LightGBM, XGBoost, Random Classifier, Logistic Regression) on a dataset with 18 attributes and 320,000 instances for heart disease prediction. The study aims to provide insights for developing accurate Heart Disease prediction models.

### Experiments

Model performances compared for the following scenarios:
- Unsampled data
- Oversampled data with outliers (Using SMOTE Oversampling Technique)
- Oversampled data without outliers (SMOTE)
- Oversampled data with outliers (Using ADASYN Oversampling Technique)
- Oversampled without outliers (ADASYN)

### Models

The four models used for this study were chosen based on the following reasons;
LightGBM and XGBoost are both gradient-boosting algorithms that are frequently used for binary classification problems like predicting heart disease diagnosis. They can handle imbalanced datasets, which is significant for this study since the dataset used is highly imbalanced across some features. These algorithms work by creating decision trees iteratively to predict the class label of the target variable. They can handle a large number of input features with high accuracy and efficiency, hence, they can handle my chosen data well.
The Random Classifier is a simple but effective ML algorithm that is suitable for classification problems like in this study. This algorithm works by randomly selecting a subset of features and creating a decision tree on the selected features. This procedure is repeated several times to create an ensemble of decision trees, which are then combined to generate the final prediction. The Random Classifier can handle noisy and imbalanced datasets and can produce accurate predictions with little computational cost.
Lastly, the Logistic RegressionCV algorithm is a variation of logistic regression that uses cross-validation to estimate the regularization parameter, which aids in reducing overfitting. Logistic regression is a linear classification algorithm that predicts the probability of an instance belonging to a particular class based on the input features. It can handle both categorical and numerical input features and is widely used for binary classification problems like this.
The modes were trained on the training sets and evaluated on the test sets.

### Findings

The evaluation metrics used are; Accuracy, Precision, Recall, F-score, and AUC

- Across all models, the initial dataset has the highest accuracy, while ADASYN with outliers has the lowest accuracy (Except for the Logistic Regression).
- The precision of all models is generally low, ranging from 0.204960 to 0.623306, indicating that many false positives are being classified as positive.
- The recall of all models is also generally low, ranging from 0.042009 to 0.815342, indicating that many true positives are being classified as false negatives.
- The F1-score is generally low, ranging from 0.078713 to 0.370072, indicating a poor balance between precision and recall.
- The AUC-ROC scores are relatively consistent across all models, ranging from 0.828601 to 0.843966.

Closer observation reveals the following:
- Both Gradient Boosting variants (LightGB and XGBoost) recorded the highest accuracies (above 83%), excluding only the Random forest model on Initial Data. However, they recorded the worst Recalls (True positive rates) on the average across all cases. Hence these models cannot be recommended.
- For all cases, only the Logistic RegressionCV model appear to have the poorest predictive accuracies (less than 80% but above 70%), but at the same time in all case, it records the highest recall (above 75%). It recorded higher recall values on the data without outliers (for both the SMOTE and ADASYN variants). The only model that came close was the Random Forest (ADASYN with outliers) at 71%.

This means that the chance that a model CORRECTLY predicts / diagnoses a patient with heart disease is higher for the Logistic RegressionCV model (ADASYN without outliers). The recall (True positive rate) is a very vital metric of the model performance as it records the number of times the model correctly predicts heart disease, for this reason, I will recommend adopting the Logistic RegressionCV model (ADASYN without outliers) for the purpose of this project application even though it has lower predictive accuracy.


