# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:41:28 2022

@author: Henry Ha
"""

# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = pd.read_csv('financial_data.csv')
dataset.info()
dataset.head()


# EDA

dataset.head()
dataset.describe()


## Cleaning Data

# Removing NaN
dataset.isna().any() # No NAs


## Histograms

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed']) # Dropping non-numeric columns

# Plotting histograms
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Correlation plot with e_signed/target variable
dataset2.corrwith(dataset['e_signed']).plot.bar(
    figsize=(20, 10), title="Correlation with E Signed", fontsize=15, rot=45, grid=True
)
# Correlation table
corr_table = pd.DataFrame(
    {'Feature': dataset2.columns.values,
     'Correlation with E Signed': dataset2.corrwith(dataset['e_signed'])}
)
print(corr_table.sort_values(by='Correlation with E Signed', ascending=False))

# Correlation matrix heatmap
corr = dataset2.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .5},
           annot=corr.round(2), fmt='.2f')
plt.show()
