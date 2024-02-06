#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:19:20 2024

@author: david
"""

'''
    Now the last part of the model, creating the train test model and outputting a confusion matrix.
    This entails splitting the dataset into a training and testing portion. 

'''
from sklearn.model_selection import train_test_split
#A couple more imports we'll use here
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

X = FourthDataFrame.drop(['deposit_new'],axis=1)
Y = FourthDataFrame['deposit_new']

#We decided a 70/30 split was enough to train the data and then test it. 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print(len(X_train))

print(len(X_test))

print("Cross-validated scores:", cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5, scoring='accuracy'))
print("Mean cross-validated score:", cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5, scoring='accuracy').mean())

model_xgb = XGBClassifier(objective='binary:logistic',learning_rate=0.1,max_depth=10,n_estimators=100)

model_xgb.fit(X_train,y_train)

Finaloutput = confusion_matrix(y_test,model_xgb.predict(X_test))
print(Finaloutput)

sns.heatmap(Finaloutput, annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True Value')
plt.show()

'''
    In summary,
    The model correctly predicted 1369 instances as positive (True Positives).
    The model correctly predicted 1489 instances as negative (True Negatives).
    The model made 310 false positive predictions (False Positives).
    The model made 177 false negative predictions (False Negatives).

    This means our accuracy, True Positives + True negatives divided by all 4 combined, resulting 
    an accuracy of roughly 85%
    
'''


