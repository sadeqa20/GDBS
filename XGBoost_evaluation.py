"""
Created on Fri Dec 13 19:28:55 2024

@author: Sadegh Asghari
"""

import numpy as np  
import xgboost as xgb  
from sklearn.model_selection import StratifiedKFold  
from sklearn import metrics  
from sklearn.metrics import balanced_accuracy_score  
from sklearn.preprocessing import LabelEncoder  

def evaluate_xgboost(X, y):  
    le = LabelEncoder()  
    y_encoded = le.fit_transform(y)  
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

    for i, (train_index, test_index) in enumerate(skf.split(X, y_encoded)):  
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]  
        # Initialize and fit XGBoost model  
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')  
        xgb_model.fit(X_train, y_train)  
        # Predict and calculate metrics  
        y_pred = xgb_model.predict(X_test)  
        accuracy[i] = metrics.accuracy_score(y_test, y_pred)  
        precision[i] = metrics.precision_score(y_test, y_pred, average='macro', zero_division=1)  
        recall[i] = metrics.recall_score(y_test, y_pred, average='macro', zero_division=1)  
        f1[i] = metrics.f1_score(y_test, y_pred, average='macro', zero_division=1)  
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