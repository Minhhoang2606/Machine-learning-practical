#pylint: disable-all
'''
Breast Cancer Classification
Data source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Author: Henry Ha
'''
# Import libraries
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.metrics import confusion_matrix # Confusion matrix
from sklearn.metrics import classification_report # Classification report
from sklearn.metrics import accuracy_score # Accuracy score
from sklearn.model_selection import GridSearchCV # GridSearchCV


# STEP 1: PROBLEM STATEMENT
'''
Details in the article: https://medium.com/@hminhhong/323-ad415293c5d4
'''
# STEP 2: IMPORTING DATA

# Loading the Breast Cancer dataset
from sklearn.datasets import load_breast_cancer # type: ignore
cancer = load_breast_cancer()
cancer['DESCR'] # type: ignore

# Creating a DataFrame
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target'])) # type: ignore
df_cancer.info() # type: ignore

# STEP #3: VISUALIZING THE DATA

# Pair plot for selected features with 'target' as hue
sns.pairplot(df_cancer, vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'], hue='target')
plt.show()

# Count plot to show the distribution of malignant and benign cases
sns.countplot(x='target', data=df_cancer, palette=['red', 'green'])
plt.xlabel('Target (0 = Malignant, 1 = Benign)')

# Annotate the count on top of each bar
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height() + 1), ha='center', va='baseline')

plt.show()

# Scatter plot for mean area vs mean smoothness
sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=df_cancer)
plt.show()

# Heatmap for feature correlations
plt.figure(figsize=(20, 10))
sns.heatmap(df_cancer.corr(), annot=True, fmt=".2f")
plt.show()

# STEP 4: MODEL TRAINING (FINDING A PROBLEM SOLUTION)

# Defining features (X) and target (y)
X = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Initializing the SVC model
svc_model = SVC()

# Training the model
svc_model.fit(X_train, y_train)

# STEP 5: EVALUATING THE MODEL

# Making predictions on the test set
y_pred = svc_model.predict(X_test)

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification report
print(classification_report(y_test, y_pred))

# STEP 6: IMPROVING THE MODEL

# Applying normalization to the training and test data
# Normalizing the training data
X_train_min = X_train.min()
X_train_range = X_train.max() - X_train_min
X_train_scaled = (X_train - X_train_min) / X_train_range

# Normalizing the test data
X_test_min = X_test.min()
X_test_range = X_test.max() - X_test_min
X_test_scaled = (X_test - X_test_min) / X_test_range

# Defining the parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Performing grid search
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train_scaled, y_train)

# Best parameters
print("Best Parameters:", grid.best_params_)

# Making predictions with the optimized model
y_pred_optimized = grid.predict(X_test_scaled)

# Evaluating the improved model
from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, y_pred_optimized))
print(f"Improved Model Accuracy: {accuracy_score(y_test, y_pred_optimized) * 100:.2f}%")

