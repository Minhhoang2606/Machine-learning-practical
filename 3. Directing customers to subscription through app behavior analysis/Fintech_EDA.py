# pylint: disable-all
"""
Created on Sat Sep 18 19:15:48 2022

@author: Henry Ha
"""
#### Importing Libraries ####

import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = pd.read_csv('new_appdata10.csv')
dataset.info()

# EDA

dataset.head(10) # Viewing the Data
dataset.describe() # Distribution of Numerical Variables

## First set of Feature cleaning
dataset["hour"] = dataset.hour.str.slice(1, 3).astype(int)

## Plotting

## Histograms
# Drop unnecessary columns for plotting
dataset2 = dataset.copy().drop(columns=['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])

# Plot histograms for numerical features
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    plt.hist(dataset2.iloc[:, i - 1], bins=np.size(dataset2.iloc[:, i - 1].unique()), color='#3F5D7D')
    plt.title(dataset2.columns.values[i - 1])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

## Correlation analysis

### Correlation with Response Variable
dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reposnse variable',
                  fontsize = 15, rot = 45,
                  grid = True)


### Compute the correlation matrix
corr = dataset2.corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) # Generate a mask for the upper triangle

### Plot the heatmap
plt.figure(figsize=(18, 15))
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
plt.title("Correlation Matrix", fontsize=20)
plt.show()

# Feature Engineering

## Convert the columns to datetime
dataset["first_open"] = pd.to_datetime(dataset["first_open"], errors='coerce')
dataset["enrolled_date"] = pd.to_datetime(dataset["enrolled_date"], errors='coerce')

# Calculating time difference between first open and enrollment in hours
dataset["difference"] = (dataset["enrolled_date"] - dataset["first_open"]).astype('timedelta64[h]')

# Plotting the distribution to confirm cutoff
plt.hist(dataset["difference"].dropna(), color='#3F5D7D')
plt.title('Distribution of Time-Since-Enrollment')
plt.xlabel('Time Since First Open (hours)')
plt.ylabel('Number of Users')
plt.show()

# Plotting the distribution within the first 500 hours
plt.hist(dataset["difference"].dropna(), bins=500, range=(0, 500), color='#3F5D7D')
plt.title('Distribution of Time-Since-Enrollment (First 500 Hours)')
plt.xlabel('Time Since First Open (hours)')
plt.ylabel('Number of Users')
plt.show()

# Plotting the distribution within the first 48 hours
plt.hist(dataset["difference"].dropna(), bins=48, range=(0, 48), color='#3F5D7D')
plt.title('Distribution of Time-Since-Enrollment (First 48 Hours)')
plt.xlabel('Time Since First Open (hours)')
plt.ylabel('Number of Users')
plt.show()

# Setting cutoff for enrollment at 48 hours
dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns=['enrolled_date', 'difference', 'first_open'])

# Display all unique values in the 'screen_list' column
unique_screens = dataset["screen_list"].str.split(',', expand=True).stack().unique()
print(unique_screens)

# Define screen categories
savings_screens = ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5", "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"]
credit_screens = ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"]
cc_screens = ["CC1", "CC1Category", "CC3"]
loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]

# Mapping screen lists to engagement funnels
dataset["SavingCount"] = dataset["screen_list"].apply(lambda x: sum(screen in x for screen in savings_screens))
dataset["CMCount"] = dataset["screen_list"].apply(lambda x: sum(screen in x for screen in credit_screens))
dataset["CCCount"] = dataset["screen_list"].apply(lambda x: sum(screen in x for screen in cc_screens))
dataset["LoansCount"] = dataset["screen_list"].apply(lambda x: sum(screen in x for screen in loan_screens))
dataset.head()

# Concatenate all screens in defined categories
all_category_screens = savings_screens + credit_screens + cc_screens + loan_screens

# Convert screen_list column to a list of screens for each entry
dataset["screen_list"] = dataset["screen_list"].apply(lambda x: x.split(','))

# Define screen categories
savings_screens = ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5", "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"]
credit_screens = ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"]
cc_screens = ["CC1", "CC1Category", "CC3"]
loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]

# Combine all category screens into one list for filtering
all_category_screens = savings_screens + credit_screens + cc_screens + loan_screens

# Filter out category screens and count remaining screens for the 'Other' feature
dataset["Other"] = dataset["screen_list"].apply(lambda screens: sum(screen not in all_category_screens for screen in screens))

# Drop the original 'screen_list' column
dataset = dataset.drop(columns=["screen_list"])

# Save the final processed dataset
dataset.to_csv("processed_appdata10.csv", index=False)


