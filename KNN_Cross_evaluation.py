"""
Created on Fri Dec 13 19:28:55 2024

@author: Sadegh Asghari
"""

import numpy as np  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import StratifiedKFold, GridSearchCV  
from sklearn import metrics  
from sklearn.metrics import balanced_accuracy_score  

def evaluate_knn_cross(X, y):  
    # Define the grid of hyperparameters  
    param_grid = {'n_neighbors': range(1, 21)}  # You can adjust the range as necessary  
    # Initialize the KNN classifier  
    knn = KNeighborsClassifier()  
    # Set up the GridSearchCV with StratifiedKFold  
    grid_search = GridSearchCV(knn, param_grid, scoring='balanced_accuracy', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    # Fit GridSearchCV on the whole dataset  
    grid_search.fit(X, y)  
    # Get the best model from grid search  
    best_knn = grid_search.best_estimator_  
    best_k = grid_search.best_params_['n_neighbors']  
    # Output the best K value found  
    print(f"Best number of neighbors (K): {best_k}")  
    # Initialize metrics arrays  
    accuracy = np.zeros(5)  
    precision = np.zeros(5)  
    recall = np.zeros(5)  
    f1 = np.zeros(5)  
    mcc = np.zeros(5)  
    balanced_acc = np.zeros(5)  
    # Perform Stratified K-Fold Cross-Validation  
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):  
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
        y_train, y_test = y[train_index], y[test_index]  
        # Fit the best model found  
        best_knn.fit(X_train, y_train)  
        # Predict and calculate metrics  
        y_pred = best_knn.predict(X_test)  
        accuracy[i] = metrics.accuracy_score(y_test, y_pred)  
        precision[i] = metrics.precision_score(y_test, y_pred, average='macro', zero_division=1)  
        recall[i] = metrics.recall_score(y_test, y_pred, average='macro', zero_division=1)  
        f1[i] = metrics.f1_score(y_test, y_pred, average='macro', zero_division=1)  
        mcc[i] = metrics.matthews_corrcoef(y_test, y_pred)  
        balanced_acc[i] = balanced_accuracy_score(y_test, y_pred)  
    # Return average metrics as a dictionary  
    return {  
        'best_k': best_k,  
        'mcc': np.mean(mcc) * 100,  
        'accuracy': np.mean(accuracy) * 100,  
        'precision': np.mean(precision) * 100,  
        'recall': np.mean(recall) * 100,  
        'f1_score': np.mean(f1) * 100,  
        'balanced_accuracy': np.mean(balanced_acc) * 100  
    }  