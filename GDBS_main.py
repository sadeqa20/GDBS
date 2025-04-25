"""
Created on Fri Dec 13 19:28:55 2024

@author: Sadegh Asghari
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans 
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import math 
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore") 
 
# Read Data Set
df = pd.read_excel('/datasets/colon.xlsx' , header=None)
#df = pd.read_excel('/datasets/CNS.xlsx' , header=None)  
#df = pd.read_csv('/datasets/GLI.csv',header=None)
#df = pd.read_csv('/datasets/SMK.csv',header=None)
#df = pd.read_excel('/datasets/Leukemia_3c.xlsx', header=None )
#df = pd.read_excel('/datasets/Covid.xlsx')
#df = pd.read_excel('/datasets/MLL.xlsx', header=None ) 
#df = pd.read_excel('/datasets/SRBCT.xlsx', header=None)

df_numpy = df.to_numpy()
df = pd.DataFrame(df_numpy)  
X = df.iloc[:, 0:df.shape[1] - 1]  
y = df.iloc[:, df.shape[1] - 1]  
X=StandardScaler().fit_transform(X)
X=pd.DataFrame(X)
alpha = np.zeros(X.shape[1])
        
##############################   (Gini Impurity)    ########################    
def gini_based_impurity (arr):
    unique_elements = np.unique(arr)
    w_values = []
    
    for current_element in unique_elements:
        if current_element not in arr:
            continue
        indices = np.where(arr == current_element)[0]
        arr_element = arr[indices[0]:indices[-1] + 1]
        s=0
        for current_element2 in unique_elements:
          n_other = np.count_nonzero(arr_element != current_element2)
          n_current = np.count_nonzero(arr_element == current_element2)
          p=(n_current / (n_other + n_current))
          if p!=0: 
             s=(p**2)+ s
        w_values.append(1-s)
    return 1-np.min(w_values)
########### (Calculate Gini Impurity)
statr_time_gini = time.time()    
for w in range(X.shape[1]):
        print(w) 
        newdf=df.sort_values(w) 
        alpha[w]=gini_based_impurity(np.array(newdf.iloc[:,newdf.shape[1]-1]))  
        
end_time_gini = time.time()
execution_time_gini = end_time_gini - statr_time_gini
        
#########################  (discretization)     #######################################     
limit= int(X.shape[1] * 0.1) 
zz=(alpha).argsort()[::-1]
Xn=zz[:int(limit)]
Xn=X[Xn]
clusters=pd.DataFrame()
matrix = Xn.to_numpy()
p = math.ceil(np.sqrt(matrix.shape[0])/2)
clusters_list = [] 
k=int(max(y))
start_time_Kmeasn = time.time() 
clusters = pd.DataFrame()  

for i in range(limit):  
    dataa = matrix[:, i].reshape(-1, 1)  
    maxnmi = 0  
    cluster_df = None  
    for k in range(int(max(y)), p+1):  
        print(f"Processing feature index: {i}, cluster count: {k}")  
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(dataa)  
        # Calculate NMI between cluster labels and true labels  
        nmi = normalized_mutual_info_score(y, kmeans.labels_)   
        if nmi > maxnmi:  
            maxnmi = nmi  
            cluster_df = pd.DataFrame(kmeans.labels_)  
    if cluster_df is not None:  
        clusters = pd.concat([clusters, cluster_df], axis=1)  

clusters.columns = range(clusters.shape[1])  
end_time_Kmeasn = time.time() 
execution_time_Kmeasn= end_time_Kmeasn - start_time_Kmeasn
nmi_result = {}
nmi_result =[]
selected_nmi = []
selected_indices = []

for i in range(limit):  
     nmi_result.append((f'{i}', nmi))   

X_matrix = pd.DataFrame(matrix)
sorted_nmi_results = dict(sorted(nmi_result, key=lambda item: item[1], reverse=True))  
max_nmi_feature_index = int(list(sorted_nmi_results.keys())[0])  
max_nmi_value = list(sorted_nmi_results.values())[0] 
selected_nmi.append(max_nmi_value)
index = max_nmi_feature_index

while len(selected_nmi) < 11 and nmi_result:   
    mean_nmi = np.mean(selected_nmi) if selected_nmi else 0    
    nmi_differences = [(index, nmi - mean_nmi) for index, nmi in nmi_result]   
    max_difference_feature = max(nmi_differences, key=lambda item: item[1])   
    index = int(max_difference_feature[0])  
    selected_nmi.append(nmi_result[index][1])  
    selected_indices.append(index)  
    nmi_result = [item for item in nmi_result if item[0] != max_difference_feature[0]]  

top_NMI_matrix = X_matrix.iloc[:,selected_indices]
top_NMI_Index = Xn.columns.tolist()
top_NMI_Indices = [top_NMI_Index[i] for i in selected_indices]
 
################
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import cross_val_score 

XX =top_NMI_matrix
label = pd.DataFrame(y)
df_new = pd.concat([XX, label], axis=1)
num_datasets = 5  
datasets = []  

################################# (Bootstrapped FSS)   #################################
model = DecisionTreeClassifier(random_state=42)
for _ in range(num_datasets):  
    bootstrap_sample = df_new.sample(n=len(df_new), replace=True)  
    datasets.append(bootstrap_sample)  

best_accuracy = 0  
best_columns = []  
accuracy_list = {}   
all_best_columns = []   
best_temp = []  
start_time = time.time()   
for dataset_index, dataset in enumerate(datasets):  
    X_new = dataset.iloc[:, :-1]  
    y_new = dataset.iloc[:, -1] 
    remaining_columns = list(range(X_new.shape[1]))  
    for i in remaining_columns:  
        subset = X_new.iloc[:, i].values.reshape(-1, 1)  
        cv_scores = cross_val_score(model, subset, y_new, cv=5)  
        accuracy = cv_scores.mean()  
        print(f'Feature {i}: Accuracy = {accuracy * 100:.2f}%')  
        if accuracy > best_accuracy:  
            best_accuracy = accuracy  
            best_columns = [i]  
            best_subset = subset 
            best_col = [i]
    print(f'\nBest initial feature: {best_columns[0]} with accuracy = {best_accuracy * 100:.2f}%')  
    remaining_columns = [i for i in range(X_new.shape[1]) if i not in best_columns]  
    best_subset = X_new.iloc[:, best_columns[0]].to_frame()  
    improved = True   
    col_best = best_columns[0]
    k=1
    best_subset_index = best_subset.shape[1]
    
    while improved and remaining_columns:  
        improved = False 
        t = col_best
        for col in remaining_columns:  
            temp_subset = pd.concat([best_subset.reset_index(drop=True), X_new.iloc[:, col].reset_index(drop=True)], axis=1)
            cv_scores = cross_val_score(model, temp_subset, y_new, cv=5)  # 5-fold cross-validation  
            accuracy = cv_scores.mean()   
            print(f'Adding feature {col}: Accuracy = {accuracy * 100:.2f}%')  
            if accuracy > best_accuracy:  
                best_accuracy = accuracy  
                col_best = col
                best_temp = temp_subset
            if col == max(remaining_columns):  
                best_columns.append(col_best)    
                best_subset = best_temp   
                if t != col_best:    
                    remaining_columns.remove(col_best)   
                    best_col.append(col_best)
                k=k+1
                print(k)
            if k == 9:
                break
            improved = True 
        best_subset_index1 = best_subset.shape[1]
        
        if best_subset_index == best_subset_index1:
            break
        else:
            best_subset_index = best_subset_index1
    final_matrix = X.iloc[:, best_col]  
    all_best_columns.append(best_col)
    
best_columns_df = pd.DataFrame(all_best_columns)  
best_columns_flattened = best_columns_df.values.flatten() 
best_columns_filtered = best_columns_flattened[~np.isnan(best_columns_flattened)]  
unique, counts = np.unique(best_columns_filtered, return_counts=True) 
feature_counts = pd.DataFrame({'Feature': unique, 'Count': counts})  
filtered_features = feature_counts[feature_counts['Count'] > 3]  
selected_features = filtered_features['Feature'].tolist()
XX.columns =[0,1,2,3,4,5,6,7,8,9]
final_X = XX.iloc[:,selected_features] 
end_time = time.time()  

print(f'\nFinal accuracy = {best_accuracy * 100:.2f}% with features: {best_col}')  
print(f'Final feature matrix:\n{final_matrix.head()}')  
time_BFS = end_time - start_time  
top_selected_Indices = [top_NMI_Indices[i] for i in selected_features]

################ (Calculate Accuracy)#####################
from decision_tree_evaluation import evaluate_decision_tree
from naive_bayes_evaluation import evaluate_naive_bayes 
from svm_rbf_evaluation import evaluate_svm_rbf
from random_forest_evaluation import  evaluate_random_forest
from KNN_evaluation import evaluate_knn
from KNN_Cross_evaluation import evaluate_knn_cross
from mlp_evaluate import evaluate_mlp
from XGBoost_evaluation import evaluate_xgboost

print('#############_________________ FULL DATA _______________ ########')
evaluate_decision_tree(X, y)
evaluate_naive_bayes(X, y)
evaluate_random_forest(X, y)
evaluate_svm_rbf(X, y) 
evaluate_mlp(X, y)
evaluate_knn(X, y)
evaluate_knn_cross(X, y)
evaluate_xgboost(X, y)

print('#############_________________ Gini Impurity _______________ ########')
evaluate_decision_tree(Xn, y)
evaluate_naive_bayes(Xn,y)
evaluate_random_forest(Xn,y)
evaluate_svm_rbf(Xn,y) 
evaluate_mlp(Xn, y)
evaluate_knn(Xn, y) 
evaluate_knn_cross(Xn,y)
evaluate_xgboost(Xn,y)

print('#############_________________ Gini Impurity Discretization _______________ ########')
evaluate_decision_tree(top_NMI_matrix, y)
evaluate_naive_bayes(top_NMI_matrix,y)
evaluate_random_forest(top_NMI_matrix,y)
evaluate_svm_rbf(top_NMI_matrix,y)
evaluate_mlp(top_NMI_matrix, y)
evaluate_knn(top_NMI_matrix, y) 
evaluate_knn_cross(top_NMI_matrix, y) 
evaluate_xgboost(top_NMI_matrix, y) 

print('#############_________________ GDBS _______________ ########')
evaluate_decision_tree(final_X, y)
evaluate_naive_bayes(final_X,y)
evaluate_random_forest(final_X,y)
evaluate_svm_rbf(final_X,y)
evaluate_mlp(final_X, y)
evaluate_knn(final_X, y)
evaluate_knn_cross(final_X, y)
evaluate_xgboost(final_X, y)

results_summary = {  
    'Algorithm': [],  
    'Dataset Size': [],  
    'MCC': [],  
    'Accuracy': [],  
    'Precision': [],  
    'Recall': [],  
    'F1 Score': [],  
    'Balanced Accuracy': [],  
    'Method': []  
}   

 
def store_results(dataset_name, algorithm_name, evaluate_function, method_name):  
    results = evaluate_function(dataset_name, y)  
    if results is None:  
        print(f"Warning: {algorithm_name} returned None.")  
        return  
    results_summary['Algorithm'].append(algorithm_name)   
    results_summary['Method'].append(method_name)  
    results_summary['Dataset Size'].append(dataset_name.shape[1])  
    results_summary['MCC'].append(results['mcc'])   
    results_summary['Accuracy'].append(results['accuracy'])   
    results_summary['Precision'].append(results['precision'])  
    results_summary['Recall'].append(results['recall'])  
    results_summary['F1 Score'].append(results['f1_score'])   
    results_summary['Balanced Accuracy'].append(results['balanced_accuracy'])   


# Evaluate FULL DATA  
print('#############_________________ FULL DATA _______________ ########')  
for algorithm in [ 'Decision Tree', 'Naive Bayes', 'Random Forest', 'SVM RBF','mlp','KNN', 'KNN_cross', 'XGBOOST']:  
    store_results(X, algorithm, eval(f'evaluate_{algorithm.lower().replace(" ", "_").replace("with_rbf_kernel", "rbf")}'), 'FULL DATA')  

# Evaluate Gini Impurity  
print('#############_________________ Gini Impurity _______________ ########')  
for algorithm in [ 'Decision Tree', 'Naive Bayes', 'Random Forest', 'SVM RBF','mlp','KNN', 'KNN_Cross','XGBOOST']: 
    store_results(Xn, algorithm, eval(f'evaluate_{algorithm.lower().replace(" ", "_").replace("with_rbf_kernel", "rbf")}'), 'Gini')  

# Evaluate Gini Impurity Discretization
print('#############_________________ Gini Impurity Discretization _______________ ########')  
for algorithm in [ 'Decision Tree', 'Naive Bayes', 'Random Forest', 'SVM RBF','mlp','KNN', 'KNN_Cross','XGBOOST']: 
    store_results(top_NMI_matrix, algorithm, eval(f'evaluate_{algorithm.lower().replace(" ", "_").replace("with_rbf_kernel", "rbf")}'), 'GD')  

# Evaluate GDBS 
print('#############_________________ GDBS _______________ ########')  
for algorithm in [ 'Decision Tree', 'Naive Bayes', 'Random Forest', 'SVM RBF', 'mlp','KNN', 'KNN_Cross','XGBOOST']: 
    store_results(final_X, algorithm, eval(f'evaluate_{algorithm.lower().replace(" ", "_").replace("with_rbf_kernel", "rbf")}'), 'GDBS')  

numeric_columns = ['Accuracy']  
results_df = pd.DataFrame(results_summary)  
results_df[numeric_columns] = results_df[numeric_columns].round(2)  

# Save all results to Excel  
results_df.to_excel('/GDBS/results_df.xlsx', index=False) 
print(f"Top_discretization_Index: {top_NMI_Indices}") 
print(f"Top_Selected_Final_Index: {top_selected_Indices}")
print(f"Time Gini Impurity: {execution_time_gini:.6f} Second") 
print(f"Time Discretization: {execution_time_Kmeasn:.6f} Second")
print(f'Time BFSS time = {time_BFS:.6f} second')