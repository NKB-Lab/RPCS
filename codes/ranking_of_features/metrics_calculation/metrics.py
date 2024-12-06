import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.utils import shuffle
#import matplotlib.pyplot as plt
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description='Train Logistic Regression, SVM, and Random Forest models with MICE imputation, saving metrics and curves data.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file.')
parser.add_argument('--output_tag', type=str, required=True, help='Tag for output files.')
args = parser.parse_args()

# Load data and replace '.' with np.nan
data = pd.read_csv(args.input_file)
data.replace(".", np.nan, inplace=True)

# Data preprocessing: drop unnecessary columns, map target labels
data = data.drop(columns=['Chromosome_Start_Ref_Alt'])
data['Status'] = data['Status'].map({'Yes': 1, 'No': 0})

# Impute missing values using MICE (IterativeImputer)
imputer = IterativeImputer(random_state=42)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split data into features and target
X = data_imputed.iloc[:, :-1]
y = data_imputed['Status']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to train
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=0),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=0),
    'SVM': SVC(kernel='linear', random_state=0, probability=True)
}

# Output metrics storage
metrics_records = []

# Train each model and store metrics
for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict probabilities and class labels for the test set
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob, average='weighted')
    auc_pr = average_precision_score(y_test, y_pred_prob, average="weighted")
    precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_pred_prob)
    #auc_pr = auc(recall_pr, precision_pr)
    f1_scr = f1_score(y_test, y_pred, average='weighted')

    # Calculate weighted and unweighted (macro) averages
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')


    
    # Store metrics in a list
    metrics_records.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision_weighted,
        "Recall": recall_weighted,
        "F1-score": f1_weighted,
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr
    })
    
    # Generate and save ROC and PR curve data
    roc_data_file = f"{args.output_tag}_{model_name}_roc_curve_data.csv"
    pr_data_file = f"{args.output_tag}_{model_name}_pr_curve_data.csv"
    
    # Generate ROC curve values and save to CSV
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_prob)
    roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Threshold': roc_thresholds})
    roc_data.to_csv(roc_data_file, index=False)
    print(f"ROC curve data for {model_name} saved to '{roc_data_file}'")
    
    # Save Precision-Recall curve data to CSV
    pr_data = pd.DataFrame({'Recall': recall_pr, 'Precision': precision_pr})
    pr_data.to_csv(pr_data_file, index=False)
    print(f"Precision-Recall curve data for {model_name} saved to '{pr_data_file}'")

# Save all metrics to a CSV file
metrics_file = f"{args.output_tag}_metrics.csv"
metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv(metrics_file, index=False)
print(f"Metrics for all models saved to '{metrics_file}'")

