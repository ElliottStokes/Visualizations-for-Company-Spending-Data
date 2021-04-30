# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:07:28 2020

@author: Elliott Stokes

@References:
    https://www.kaggle.com/fabiendaniel/predicting-flight-delays-tutorial

This file will deal with the preprocessing of the data including:
    cleaning
    Dealing with missing values
    Validating datasets
"""

import pandas as pd
import os
import re

dataset = pd.DataFrame()

for location in os.listdir('../DataSets'):
    print(location)
    for filename in os.listdir('../DataSets/{}'.format(location)):
        if filename.endswith('.csv'):
            print(filename)
            df = pd.read_csv('../DataSets/{}/{}'.format(location, filename), encoding= 'unicode_escape')
            
            # Rename the columns with inconsistencies and padding
            amountColName = [col for col in df.columns if 'Amount' in col]
            df = df.rename(columns={'Expense area': 'Expense Area', amountColName[0]: 'Amount'})
            
            # Convert the date column to the pandas datetime format
            df['Date'] = pd.to_datetime(df.Date)
            # df['Date'] = df['Date'].dt.date
            
            # Convert the Amount column to type float
            df['Amount'] = df['Amount'].astype(str)
            df['Amount'] = df['Amount'].str.replace(',', '', regex = True)
            df['Amount'] = df['Amount'].str.replace('£', '', regex = True)
            df['Amount'] = df['Amount'].astype(float)
            
            # Add source location to dataframe
            df['Group'] = location
            
            dataset = dataset.append(df[['Date', 'Group', 'Expense Type', 'Expense Area', 'Supplier', 'Amount']], ignore_index = True)

print('dataset dimensions:', dataset.shape)

# Examine the completeness of the dataset

missing_df = dataset.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(dataset.shape[0]-missing_df['missing values'])/dataset.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)

print(missing_df)

def get_stats(group):
    return {'count': group.count(), 'min': group.min(), 'max': group.max(), 'mean': group.mean()}

amount_stats = dataset['Amount'].groupby([dataset['Date'].dt.strftime('%Y')]).apply(get_stats).unstack()
amount_stats = amount_stats.sort_values('mean')
print(amount_stats)

ccg_stats = dataset['Amount'].groupby([dataset['Group']]).apply(get_stats).unstack()
ccg_stats = ccg_stats.sort_values('mean')
print(ccg_stats)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (15, 15))
sns.set()
sns.set(style = "darkgrid")
ax = sns.countplot(y = dataset['Expense Type'], 
                    data = dataset, 
                    order = dataset['Expense Type'].value_counts().iloc[:20].index)
ax.set_title('Top 20 most frequent expense types')
plt.show()

plt.figure(figsize = (20, 5))
ax2 = sns.stripplot(y="Group", x="Amount", size = 4, data = dataset.loc[:, ['Group', 'Amount']], linewidth = 1, jitter = True)
plt.setp(ax2.get_xticklabels(), fontsize = 14)
plt.setp(ax2.get_yticklabels(), fontsize = 14)
ax2.set_xticklabels(['{:2.0f}'.format(*[int(y) for y in divmod(x,100)]) for x in ax2.get_xticks()])
plt.xlabel('Amount')
ax2.yaxis.label.set_visible(False)
plt.show()

amount_type = lambda x:((0,1)[x > 50000],2)[x > 1000000]
dataset['Amount level'] = dataset['Amount'].apply(amount_type)

plt.figure(1, figsize = (10,7))
ax3 = sns.countplot(y = "Group", hue = "Amount level", data=dataset)
L = plt.legend()
L.get_texts()[0].set_text('Small (Amount < £50000)')
L.get_texts()[1].set_text('Medium (£50000 < Amount < £1000000)')
L.get_texts()[2].set_text('Large (Amount > £1000000)')
plt.show()
