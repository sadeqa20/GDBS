# GDBS
GDBS: A hybrid feature selection method for small-sample, high-dimensional datasets

Python code guide and documentation: GDBS

Please refer to https://github.com/hnematzadeh/EMC-DWES/tree/main/Datasets for the datasets used.

1. Import Libraries and Prepare Data

Key Libraries: NumPy, Pandas, Scikit-learn (for data processing and modeling).

Data Loading:

Reads data from Excel/CSV files (multiple datasets are commented out).

Example: df = pd.read_excel('/datasets/colon.xlsx', header=None)

Preprocessing:

Separates features (X) and labels (y).

Standardizes features using StandardScaler().

2. Gini Impurity Calculation

Goal: Rank features based on their ability to separate classes.

Function gini_based_impurity(arr):

Computes Gini impurity for a sorted feature array.

Lower impurity = Higher feature importance.

Process:

Sorts data by each feature and calculates its Gini impurity.

Selects the top 10% of features (Xn) with the lowest impurity.

Output: A subset of high-impact features.

3. Feature Discretization via K-Means Clustering

Goal: Convert continuous features to discrete values for improved model performance.

Steps:

• For each selected feature:

 - Reshapes the feature into a 2D array for clustering.  
 
 - Tests cluster counts from the number of classes (`max(y)`) to `ceil(sqrt(n_samples)/2)`.  
 
 - Normalized Mutual Information (NMI) is used to evaluate the alignment between cluster labels and true labels.  
 
 - Retains the cluster configuration with the highest NMI.  
• Aggregates discretized features into a new matrix (clusters).

4. Bootstrapped Feature Subset Selection (FSS)

Goal: Further refine features using bootstrap sampling and forward selection.

Steps:

• Bootstrap Sampling: Generates 5 bootstrapped datasets.

• Forward Selection:

 - Starts with the best single feature (highest cross-validation accuracy).  
 
 - Iteratively adds features that maximize accuracy.  
 
 - Uses a Decision Tree classifier for evaluation.  
• Aggregate Results:

 - Features selected in >3 bootstrap samples are retained.  
Output: Final feature subset (final_X).

5. Model Evaluation

- Classifiers used:

Decision Tree, Naive Bayes, Random Forest, SVM (RBF kernel), MLP, KNN, XGBoost.

- Datasets Evaluated:

Full dataset (all features).

Gini-selected features (Xn).

Discretized features (top_NMI_matrix).

Final GDBS-selected features (final_X).

Metrics Reported:

Accuracy, MCC, Precision, Recall, F1 Score, Balanced Accuracy.

6. Save Results

Results Summary:

-Metrics for all combinations of algorithms and datasets are stored in results_df.xlsx.

Key Outputs:

Indices of top discretized features (top_NMI_Indices).

Final selected features (top_selected_Indices).

Execution times for Gini, discretization, and FSS steps.

Key Notes

Designed for High-Dimensional Data (e.g., genomic data).

Flexibility: Adjust parameters like limit (top feature percentage), p (max clusters), or num_datasets (bootstrap samples).

Reproducibility: Uses random_state=42 for deterministic results.

Suppressed Warnings: Non-critical warnings are ignored to ensure cleaner output.

To run the code:

Ensure dataset paths are correct.

Install required libraries (scikit-learn, pandas, numpy, xgboost, etc.).

Uncomment the desired dataset line (e.g., /datasets/colon.xlsx).
