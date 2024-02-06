#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:07:10 2024

@author: david
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

bank['deposit'].groupby(bank['deposit']).count()

newData = bank.copy(deep = True)
newData.head()

print(bank.columns)

#print(newData.columns)

'''
    This can be used as correlation between defaulting and the number of deposits made
    after looking, typically wether one person deposits or not the number of default 
    loans is less than the loans that don't. However the number of defaulted loans is higher 
    in the people who made no deposits. This could be due to numerous things but it is a 
    key note of loan succession. 
'''


newData.groupby(['deposit','default']).size()

'''
    From here on out we'll be focusing on featuring engineering, we already handled dropping
    the Nan values so now we'll go into looking for outliers and narrowing down what number of data
    we'll use in our model

'''
newData.drop(['default'],axis=1, inplace=True)

print(newData.groupby(['deposit','pdays']).size())

sns.boxplot(x='deposit', y='pdays', data=newData)
plt.title('Boxplot of pdays grouped by deposit')
plt.show()

'''
    Here we can see that we get a -1 at 4940 of the no positions and to be sure of any other
    instances we graphed it and saw the no's were practically nothing and the average was hovering around the 0,
    so we'll classify this column as an outlier and drop it

'''

newData.drop(['pdays'],axis=1, inplace=True)

# remove outliers in feature balance...
newData.groupby(['deposit','balance'],sort=True)['balance'].count()

#This column won't be removed because it's an even spread for different answers,
#we want variance

newData.groupby('age',sort=True)['age'].count()
'''
    These values are good, there are no "out of the ordinary" values, such that the ages seem normal.
    we probably won't use them but at least we know theres variance and no outliers. 
    This ranges from 18 - 95 years old.

'''


newData.groupby(['deposit','campaign'],sort=True)['campaign'].count()
'''
    These have variance so there's no need to eliminate the column, HOWEVER, the no deposite section of the
    campaign jumps from 33, into larger numbers, which is a key sign of outliers, especially since the yes
    portion of the campaign stops at 32, so for simplicity sake we'll treat 32 as the max and eliminate 
    anything beyond as outliers. 

'''

#Establishing the new data frame with all outliers eliminated and sorted. 
ThirdDataFrame = newData[newData['campaign'] < 32]

#Making sure everything moved over
ThirdDataFrame.head()

ThirdDataFrame.groupby(['deposit','previous'],sort=True)['previous'].count()

'''
    Upon looking at this we see that the numbers don't correlate too well as the previous column gets higher,
    after looking the numbers take a large spike around the 28-30 range for both no and yes deposits.
    So we figured a good spot to cut off outliers was 29, its right in the middle of 28 and 30, and at 29 it's
    a perfect 24 numbers for both no and yes deposites.

'''
FourthDataFrame = ThirdDataFrame[ThirdDataFrame['previous'] < 31]

'''
    As the last part of feature engineering, we need to mask column data and create dummy variables. 
    We'll separate the categorical features from before and the remaining columns that are made up of 
    phrases like yes and no

'''

categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
# Use get_dummies on categorical columns and drop the original columns
FourthDataFrame = pd.get_dummies(FourthDataFrame, columns=categorical_cols, prefix=categorical_cols, prefix_sep='_', drop_first=True, dummy_na=False)
FourthDataFrame.head()

remaining_cols = ['housing', 'deposit', 'loan']

#As an experiment we tried using the lambda function
FourthDataFrame[remaining_cols] = FourthDataFrame[remaining_cols].apply(lambda col: col.apply(lambda x: 1 if x == 'yes' else 0))

#One liner for loop
new_column_names = [col + '_new' for col in remaining_cols]
FourthDataFrame.rename(columns=dict(zip(remaining_cols, new_column_names)), inplace=True)

FourthDataFrame.head()
