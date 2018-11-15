#Importing Required Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
#Import the Dataset
dataset=pd.read_csv("winequality_red.csv")
#head() Information Of Wine Dataset
print (dataset.head())
#Separating The Data Into Features And Labels
y = dataset.quality
X = dataset.drop('quality', axis=1)
#Splitting Into Test And Train Data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print(X_train.head())
#Train Data Preprocessing
X_train_scaled = preprocessing.scale(X_train)
print (X_train_scaled)
#Train Data Preprocessing
clf=tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
#to check how efficiently your algorithm is predicting the label 
confidence = clf.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)
#prediction of data
y_pred = clf.predict(X_test)
#Comparing The Predicted And Expected Labels
#converting the numpy array to list
x=np.array(y_pred).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(5):
    print (i)
    
#printing first five expectations
print("\nThe expectation:\n")
print (y_test.head())