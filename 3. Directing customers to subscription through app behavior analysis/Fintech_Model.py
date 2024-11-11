# pylint: disable-all
"""
Created on Sat Sep 18 19:15:48 2022

@author: Henry Ha
"""

# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import time

dataset = pd.read_csv('processed_appdata10.csv')

# Data Pre-Processing

## Splitting Independent and Response Variables
response = dataset["enrolled"]
dataset = dataset.drop(columns="enrolled")

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size = 0.2, random_state = 0)

# Removing Identifiers
train_identity = X_train['user']
X_train = X_train.drop(columns = ['user'])
test_identity = X_test['user']
X_test = X_test.drop(columns = ['user'])

# Feature Scaling
# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training features, then transform the testing features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building

# Initialize logistic regression model with L1 regularization
model = LogisticRegression(penalty='l1', solver='saga', random_state=0)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Subscriber", "Subscriber"], yticklabels=["Non-Subscriber", "Subscriber"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

# Perform 10-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std() * 2:.2f}")

# Model Tuning ####

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import time

# Initialize logistic regression model
classifier = LogisticRegression(random_state=0)

### Grid Search - Round 1 ###

# Define broad hyperparameter grid for Round 1
param_grid_round1 = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Set up GridSearchCV
grid_search1 = GridSearchCV(estimator=classifier, param_grid=param_grid_round1, scoring='accuracy', cv=10, n_jobs=-1)

# Run Grid Search and time the process
t0 = time.time()
grid_search1.fit(X_train, y_train)
t1 = time.time()
print(f"Round 1 took {t1 - t0:.2f} seconds")

# Capture the best parameters from Round 1
best_params_round1 = grid_search1.best_params_
print(f"Best Parameters from Round 1: {best_params_round1}")

### Grid Search - Round 2 ###

# Define focused hyperparameter grid based on Round 1 results
param_grid_round2 = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 0.5, 0.9, 1, 2, 5]  # Narrowed range around the best values from Round 1
}

# Set up GridSearchCV for Round 2
grid_search2 = GridSearchCV(estimator=classifier, param_grid=param_grid_round2, scoring='accuracy', cv=10, n_jobs=-1)

# Run Grid Search and time the process
t0 = time.time()
grid_search2.fit(X_train, y_train)
t1 = time.time()
print(f"Round 2 took {t1 - t0:.2f} seconds")

# Capture the best parameters and model from Round 2
best_params = grid_search2.best_params_
best_model = grid_search2.best_estimator_

print(f"Best Parameters from Round 2: {best_params}")


#### End of Model ####



