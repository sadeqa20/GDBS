"""
Created on Fri Dec 13 19:28:55 2024

@author: Sadegh Asghari
"""
# naive_bayes_evaluation.py  

import numpy as np  
from sklearn.model_selection import StratifiedKFold  
from sklearn.naive_bayes import GaussianNB  
from sklearn import metrics  
from sklearn.metrics import balanced_accuracy_score  

def evaluate_naive_bayes(X, y):  
    # Initialize metrics arrays
    splite = 5
    s = np.zeros(splite)  
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
        # Initialize and fit Naive Bayes classifier  
        nb = GaussianNB()  
        nb.fit(X_train, y_train)  
        # Predict and calculate metrics  
        s[i] = nb.score(X_test, y_test)  
        y_pred = nb.predict(X_test)  
        precision[i] = metrics.precision_score(y_test, y_pred, average='macro',zero_division=1)  
        recall[i] = metrics.recall_score(y_test, y_pred, average='macro',zero_division=1)  
        f1[i] = metrics.f1_score(y_test, y_pred, average='macro',zero_division=1)  
        mcc[i] = metrics.matthews_corrcoef(y_test, y_pred)   
        balanced_acc[i] = balanced_accuracy_score(y_test, y_pred)  
    # Return average metrics as a dictionary  
    return {  
        'mcc': np.mean(mcc) * 100,  
        'accuracy': np.mean(s) * 100,  
        'precision': np.mean(precision) * 100,  
        'recall': np.mean(recall) * 100,  
        'f1_score': np.mean(f1) * 100,  
        'balanced_accuracy': np.mean(balanced_acc) * 100  
    }