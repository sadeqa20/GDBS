"""
Created on Fri Dec 13 19:28:55 2024

@author: Sadegh Asghari
"""


import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold  
from sklearn import metrics  
from sklearn.metrics import balanced_accuracy_score  

def evaluate_random_forest(X, y):  
    # Initialize metrics arrays 
    splite = 5
    accuracy = np.zeros(splite)  
    precision = np.zeros(splite)  
    recall = np.zeros(splite)  
    f1 = np.zeros(splite)  
    mcc = np.zeros(splite)  
    balanced_acc = np.zeros(splite)  
    # Perform Stratified K-Fold Cross-Validation  
    skf = StratifiedKFold(n_splits=splite, shuffle=True, random_state=42)    

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):  
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
        y_train, y_test = y[train_index], y[test_index]  
        # Initialize and fit Random Forest model  
        rf = RandomForestClassifier(random_state=42)  
        rf.fit(X_train, y_train)  
        # Predict and calculate metrics  
        y_pred = rf.predict(X_test)  
        accuracy[i] = metrics.accuracy_score(y_test, y_pred)  
        precision[i] = metrics.precision_score(y_test, y_pred, average='macro',zero_division=1)  
        recall[i] = metrics.recall_score(y_test, y_pred, average='macro',zero_division=1)  
        f1[i] = metrics.f1_score(y_test, y_pred, average='macro',zero_division=1)  
        mcc[i] = metrics.matthews_corrcoef(y_test, y_pred)   
        balanced_acc[i] = balanced_accuracy_score(y_test, y_pred)  
    # Return average metrics as a dictionary  
    return {  
        'mcc': np.mean(mcc) * 100,  
        'accuracy': np.mean(accuracy) * 100,  
        'precision': np.mean(precision) * 100,  
        'recall': np.mean(recall) * 100,  
        'f1_score': np.mean(f1) * 100,  
        'balanced_accuracy': np.mean(balanced_acc) * 100  
    }



"""
import numpy as np  
# import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold  
from sklearn import metrics  
from sklearn.metrics import balanced_accuracy_score  

def evaluate_random_forest(X, y):  
    # Initialize metrics arrays  
    accuracy = np.zeros(5)  
    precision = np.zeros(5)  
    recall = np.zeros(5)  
    f1 = np.zeros(5)  
    mcc = np.zeros(5)  
    balanced_acc = np.zeros(5)  

    # Perform Stratified K-Fold Cross-Validation  
    skf = StratifiedKFold(n_splits=5)  

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):  
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
        y_train, y_test = y[train_index], y[test_index]  

        # Initialize and fit Random Forest model  
        rf = RandomForestClassifier(random_state=42)  
        rf.fit(X_train, y_train)  
        
        # Predict and calculate metrics  
        y_pred = rf.predict(X_test)  
        accuracy[i] = metrics.accuracy_score(y_test, y_pred)  
        precision[i] = metrics.precision_score(y_test, y_pred, average='micro')  
        recall[i] = metrics.recall_score(y_test, y_pred, average='micro')  
        f1[i] = metrics.f1_score(y_test, y_pred, average='micro')  
        mcc[i] = metrics.matthews_corrcoef(y_test, y_pred)   
        balanced_acc[i] = balanced_accuracy_score(y_test, y_pred)  

    # Print average metrics  
    print('===============================Random Forest Classifier:')  
  #  print('MCC =', round(np.mean(mcc), 2))
    print(f'MCC = {np.mean(mcc) * 100:.2f}%')
  #  print('Accuracy =', round(np.mean(accuracy), 2))  
    print(f'Accuracy = {np.mean(accuracy) *100:.2f}%')
  #  print('Precision =', round(np.mean(precision), 2))  
    print(f'Precision = {np.mean(precision) * 100:.2f}%')  
  #  print('Recall =', round(np.mean(recall), 2))  
    print(f'Recall = {np.mean(recall) * 100:.2f}%')  
  #  print('F1 Score =', round(np.mean(f1), 2))  
    print(f'F1 Score = {np.mean(f1) * 100:.2f}%') 
  #  print('Balanced Accuracy =', round(np.mean(balanced_acc), 2))  
    print(f'Balanced Accuracy {np.mean(balanced_acc) * 100:.2f}%')  
    
"""
# Example usage  
# Assuming X_reduced and y are defined  
# evaluate_random_forest(X_reduced, y)