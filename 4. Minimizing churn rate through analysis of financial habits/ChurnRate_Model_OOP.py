"""
Created on Sun Aug 26 18:46:58 2022

@author: Henry Ha
"""
# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load Data
dataset = pd.read_csv('processed_churn_data.csv')

# Define the independent variables X and dependent variable y
X = dataset.drop(columns=['churn'])
y = dataset['churn']

# Apply one-hot encoding to categorical features
X = pd.get_dummies(X, columns=['housing', 'payment_type', 'zodiac_sign'], drop_first=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the training set using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['age', 'withdrawal', 'purchases_partners', 'purchases', 'reward_rate', 'registered_phones']
X_train_smote[numerical_features] = scaler.fit_transform(X_train_smote[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Define the ModelEvaluator class for repeated steps
class ModelEvaluator:
    def __init__(self, model, X, y, cv=10):
        """
        Initialize the ModelEvaluator with a classifier, features, target, and cross-validation folds.
        """
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def k_fold_cross_validation(self):
        """
        Perform K-Fold Cross-Validation and print the mean accuracy.
        """
        accuracies = cross_val_score(estimator=self.model, X=self.X, y=self.y, cv=self.cv)
        print("Mean Accuracy: {:.3f} (+/- {:.3f})".format(accuracies.mean(), accuracies.std() * 2))
        return accuracies

    def calculate_confusion_matrix(self):
        """
        Calculate and print the confusion matrix and other performance metrics.
        """
        y_pred = cross_val_predict(self.model, self.X, self.y, cv=self.cv)
        conf_matrix = confusion_matrix(self.y, y_pred)

        # Extract and display metrics
        accuracy = accuracy_score(self.y, y_pred)
        precision = precision_score(self.y, y_pred)
        recall = recall_score(self.y, y_pred)
        f1 = f1_score(self.y, y_pred)

        print("Confusion Matrix:\n", conf_matrix)
        print("Accuracy: {:.2f}".format(accuracy))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1 Score: {:.2f}".format(f1))

        return conf_matrix, accuracy, precision, recall, f1

# Initialize the logistic regression model
classifier = LogisticRegression()

# Evaluate the original model
print("Original Model Evaluation:")
evaluator = ModelEvaluator(classifier, X_train_smote, y_train_smote)
evaluator.k_fold_cross_validation()
evaluator.calculate_confusion_matrix()

# Apply Recursive Feature Elimination (RFE)
rfe = RFE(estimator=classifier, n_features_to_select=20)
X_train_rfe = rfe.fit_transform(X_train_smote, y_train_smote)
X_test_rfe = rfe.transform(X_test)

# Training the Model with Selected Features
classifier_rfe = LogisticRegression(random_state=42)
classifier_rfe.fit(X_train_rfe, y_train_smote)

# Evaluate the RFE model
print("\nRFE Model Evaluation:")
rfe_evaluator = ModelEvaluator(classifier, X_train_rfe, y_train_smote)
rfe_evaluator.k_fold_cross_validation()
rfe_evaluator.calculate_confusion_matrix()
