# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:41:44 2022

@author: Henry Ha
"""
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


# Data Preprocessing

dataset = pd.read_csv('financial_data.csv')

# Feature Engineering

# Dropping 'months_employed' due to lack of meaningful data
dataset = dataset.drop(columns=['months_employed'])

# Creating a unified 'personal_account_months' feature
dataset['personal_account_months'] = dataset['personal_account_y'] * 12 + dataset['personal_account_m']

# Dropping the original columns to avoid redundancy
dataset = dataset.drop(columns=['personal_account_y', 'personal_account_m'])

# One-hot encoding 'pay_schedule' and dropping one level to avoid dummy variable trap
dataset = pd.get_dummies(dataset, columns=['pay_schedule'], drop_first=True)

# Separating the target variable from the features
X = dataset.drop(columns=['e_signed', 'entry_id'])  # Features
y = dataset['e_signed']  # Target

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting into Train and Test Set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

#### Model Building ####

### Comparing Models
class ModelEvaluator:
    def __init__(self, model, model_name):
        """
        Initialize a ModelEvaluator with a model and a name to be used in
        the report.

        Parameters
        ----------
        model : object
            A scikit-learn model instance.
        model_name : str
            A string indicating the name of the model.
        """
        self.model = model
        self.model_name = model_name

    def train(self, X_train, y_train):
        """
        Train the model with the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training data.
        y_train : array-like of shape (n_samples,)
            The target values.
        """

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model with the test data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The test data.
        y_test : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        accuracy : float
            The accuracy of the model.
        precision : float
            The precision of the model.
        recall : float
            The recall of the model.
        f1 : float
            The F1 score of the model.
        """
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Display the results
        print(f"{self.model_name} Performance:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # Display confusion matrix
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Return results for further analysis if needed
        return accuracy, precision, recall, f1

# Define models with names
log_reg = ModelEvaluator(LogisticRegression(penalty='l1', solver='liblinear', random_state=0), "Logistic Regression")
svm_linear = ModelEvaluator(SVC(kernel='linear', random_state=0), "SVM (Linear Kernel)")
svm_rbf = ModelEvaluator(SVC(kernel='rbf', random_state=0), "SVM (RBF Kernel)")
rf = ModelEvaluator(RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0), "Random Forest")

# Train and evaluate each model
log_reg.train(X_train, y_train)
log_reg.evaluate(X_test, y_test)

svm_linear.train(X_train, y_train)
svm_linear.evaluate(X_test, y_test)

svm_rbf.train(X_train, y_train)
svm_rbf.evaluate(X_test, y_test)

rf.train(X_train, y_train)
rf.evaluate(X_test, y_test)


# First round of Grid Search with a broader range
rf_param_grid_1 = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=0)

# First round of Grid Search with 10-fold Cross-Validation
# cv=10 specifies 10-fold cross-validation, meaning the data will be split into 10 parts.
# Each part will be used as a validation set once, while the other 9 parts are used for training.
# This process is repeated 10 times for each combination of hyperparameters in the grid.
rf_grid_search_1 = GridSearchCV(estimator=rf, param_grid=rf_param_grid_1, cv=10, scoring='accuracy', n_jobs=-1)
rf_grid_search_1.fit(X_train, y_train)

# Extract best parameters from the first round
best_params_1 = rf_grid_search_1.best_params_

# Refined parameter grid for the second round based on first round results
# Refined parameter grid for the second round based on first round results
rf_param_grid_2 = {
    'n_estimators': [best_params_1['n_estimators'] - 10, best_params_1['n_estimators'], best_params_1['n_estimators'] + 10],
    'max_depth': [best_params_1['max_depth'] - 2, best_params_1['max_depth'], best_params_1['max_depth'] + 2] if best_params_1['max_depth'] else [None],
    'min_samples_leaf': [max(1, best_params_1['min_samples_leaf'] - 1), best_params_1['min_samples_leaf'], best_params_1['min_samples_leaf'] + 1],
    'criterion': [best_params_1['criterion']]
}

# Second round of Grid Search
rf_grid_search_2 = GridSearchCV(estimator=rf, param_grid=rf_param_grid_2, cv=10, scoring='accuracy', n_jobs=-1)
rf_grid_search_2.fit(X_train, y_train)

# Output the best parameters and best cross-validated accuracy
print("Best Parameters for Random Forest:", rf_grid_search_2.best_params_)
print("Best Cross-Validation Accuracy:", rf_grid_search_2.best_score_)


# Use the best Random Forest model from Grid Search to make predictions on the test set
y_pred = rf_grid_search_2.best_estimator_.predict(X_test)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Generate the classification report for precision, recall, F1-score, and accuracy
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)







