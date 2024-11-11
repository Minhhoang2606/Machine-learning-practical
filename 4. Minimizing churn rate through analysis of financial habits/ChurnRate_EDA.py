# pylint: disable-all
"""
Created on Sun Aug 26 18:46:58 2022

@author: Henry Ha
"""

# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('churn_data.csv') # Users who were 60 days enrolled, churn in the next 30
dataset.info()

dataset.isna().sum()

# Dropping columns with high missing values
dataset.drop(columns=['credit_score', 'rewards_earned'], inplace=True)

# Dropping rows with missing values in 'age'
dataset = dataset[dataset['age'].notna()]

# List all columns with categorical features or 'object' data type
object_cols = dataset.select_dtypes(include=['object']).columns
print("Columns with categorical features or 'object' data type:", object_cols)

# EDA

## Plotting histograms for numerical features
import matplotlib.pyplot as plt

# List of numerical features excluding 'user'
numerical_features = ['churn', 'age', 'deposits', 'withdrawal', 'purchases_partners',
                      'purchases', 'cc_taken', 'cc_recommended', 'cc_disliked', 'cc_liked',
                      'cc_application_begin', 'app_downloaded', 'web_user', 'app_web_user',
                      'ios_user', 'android_user', 'registered_phones', 'waiting_4_loan',
                      'cancelled_loan', 'received_loan', 'rejected_loan', 'left_for_two_month_plus',
                      'left_for_one_month', 'reward_rate', 'is_referred']

# Number of columns and rows for the layout
n_cols = 5
n_rows = (len(numerical_features) + n_cols - 1) // n_cols  # Calculate required rows

# Create the figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
fig.suptitle("Distribution of Numerical Features", fontsize=20)

# Flatten axes array for easier iteration
axes = axes.flatten()

# Plot each feature in a separate subplot
for i, feature in enumerate(numerical_features):
    if feature in dataset.columns:  # Ensure the feature exists in the dataset
        axes[i].hist(dataset[feature].dropna(), bins=20, edgecolor='black')  # Drop NA values for cleaner plots
        axes[i].set_title(feature)
    else:
        axes[i].text(0.5, 0.5, f"{feature} not found", ha='center', va='center')
        axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the main title
plt.show()


# Bar plot for categorical features

# List of categorical features
categorical_features = ['housing', 'is_referred', 'app_downloaded', 'web_user', 'app_web_user',
                        'ios_user', 'android_user', 'registered_phones', 'payment_type',
                        'waiting_4_loan', 'cancelled_loan', 'received_loan', 'rejected_loan',
                        'zodiac_sign', 'left_for_two_month_plus', 'left_for_one_month']

# Number of rows and columns for the table layout (e.g., 5 rows x 4 columns)
n_rows = 5
n_cols = 4

# Create a figure with a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
fig.suptitle("Pie Chart Distributions", fontsize=20)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Loop through each feature and plot on the corresponding subplot
for i, feature in enumerate(categorical_features):
    values_counts = dataset[feature].value_counts()
    axes[i].pie(values_counts, labels=values_counts.index, autopct='%1.1f%%', startangle=140)
    axes[i].set_title(feature)

# Turn off any remaining empty subplots
for j in range(i + 1, n_rows * n_cols):
    axes[j].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
plt.show()


## Histograms
## Plotting histograms for numerical features
import matplotlib.pyplot as plt

# List of numerical features excluding 'user'
numerical_features = ['churn', 'age', 'deposits', 'withdrawal', 'purchases_partners',
                      'purchases', 'cc_taken', 'cc_recommended', 'cc_disliked', 'cc_liked',
                      'cc_application_begin', 'app_downloaded', 'web_user', 'app_web_user',
                      'ios_user', 'android_user', 'registered_phones', 'waiting_4_loan',
                      'cancelled_loan', 'received_loan', 'rejected_loan', 'left_for_two_month_plus',
                      'left_for_one_month', 'reward_rate', 'is_referred']

# Number of columns and rows for the layout
n_cols = 5
n_rows = (len(numerical_features) + n_cols - 1) // n_cols  # Calculate required rows

# Create the figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
fig.suptitle("Distribution of Numerical Features", fontsize=20)

# Flatten axes array for easier iteration
axes = axes.flatten()

# Plot each feature in a separate subplot
for i, feature in enumerate(numerical_features):
    if feature in dataset.columns:  # Ensure the feature exists in the dataset
        axes[i].hist(dataset[feature].dropna(), bins=20, edgecolor='black')  # Drop NA values for cleaner plots
        axes[i].set_title(feature)
    else:
        axes[i].text(0.5, 0.5, f"{feature} not found", ha='center', va='center')
        axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the main title
plt.show()

# List of selected features
selected_features = ['age', 'deposits', 'withdrawal', 'purchases',
                     'purchases_partners', 'cc_taken', 'cc_recommended', 'app_downloaded',
                     'web_user', 'reward_rate', 'registered_phones']

# Calculate correlations values for selected features
correlations = dataset[selected_features + ['churn']].corr()['churn'][:-1]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
correlations.plot(kind='bar', color=['#1f77b4' if corr > 0 else '#ff7f0e' for corr in correlations])
plt.title('Feature Correlation with Churn')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.axhline(0, color='grey', linewidth=0.5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

## Exploring Uneven Features
dataset[dataset.waiting_4_loan == 1].churn.value_counts()
dataset[dataset.cancelled_loan == 1].churn.value_counts()
dataset[dataset.received_loan == 1].churn.value_counts()
dataset[dataset.rejected_loan == 1].churn.value_counts()
dataset[dataset.left_for_one_month == 1].churn.value_counts()



## Correlation with Response Variable
dataset.drop(columns = ['housing', 'payment_type',
                         'registered_phones', 'zodiac_sign']
    ).corrwith(dataset.churn).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 15, rot = 45,
              grid = True)


# Compute the correlation matrix
corr_matrix = dataset.corr(numeric_only=True)

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=1)
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Dropping highly correlated features
dataset = dataset.drop(['deposits', 'app_web_user'], axis=1)
dataset.info()


## Note: Although there are somewhat correlated fields, they are not colinear
## These feature are not functions of each other, so they won't break the model
## But these feature won't help much either. Feature Selection should remove them.

dataset.to_csv('processed_churn_data.csv', index = False)
