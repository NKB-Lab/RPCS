import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve

def apply_model_to_blind_data(input_file, model_filename, output_tag):
    # Load the trained model
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    
    # Load the blind data
    blind_data = pd.read_csv(input_file)
    
    # Replace '.' with NaN in blind data
    blind_data.replace('.', np.nan, inplace=True)
    
    # Extract features (excluding the identifier and label columns)
    X_blind = blind_data.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')
    
    # Initialize the imputer with MICE and apply to the blind data
    imputer = IterativeImputer(max_iter=10, random_state=0)
    X_blind_imputed = pd.DataFrame(imputer.fit_transform(X_blind), columns=X_blind.columns)


    # Predict the probabilities and labels for blind data
    y_pred_prob = model.predict_proba(X_blind_imputed)[:, 1]
    y_pred = model.predict(X_blind_imputed)

    # Calculate metrics
    accuracy = accuracy_score(blind_data.iloc[:, -1].apply(lambda x: 1 if x == "Yes" else 0), y_pred)
    precision = precision_score(blind_data.iloc[:, -1].apply(lambda x: 1 if x == "Yes" else 0), y_pred)
    recall = recall_score(blind_data.iloc[:, -1].apply(lambda x: 1 if x == "Yes" else 0), y_pred)
    f1 = f1_score(blind_data.iloc[:, -1].apply(lambda x: 1 if x == "Yes" else 0), y_pred)
    auc_roc = roc_auc_score(blind_data.iloc[:, -1].apply(lambda x: 1 if x == "Yes" else 0), y_pred_prob)
    
    # Calculate Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(blind_data.iloc[:, -1].apply(lambda x: 1 if x == "Yes" else 0), y_pred_prob)
    auc_pr = auc(recall_vals, precision_vals)

    # Print metrics
    print("Evaluation on blind data:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    
    
    # Generate and save ROC and PR curve data
    roc_data_file = f"{output_tag}_auc-roc_plot_data.csv"
    pr_data_file = f"{output_tag}_auc-pr_plot_data.csv"
    
    # Generate ROC curve values and save to CSV
    fpr, tpr, roc_thresholds = roc_curve(blind_data.iloc[:, -1].apply(lambda x: 1 if x == "Yes" else 0), y_pred_prob)
    roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Threshold': roc_thresholds})
    roc_data.to_csv(roc_data_file, index=False)
    print(f"ROC curve saved to '{roc_data_file}'")
    
    # Save Precision-Recall curve data to CSV
    pr_data = pd.DataFrame({'Recall': recall_vals, 'Precision': precision_vals})
    pr_data.to_csv(pr_data_file, index=False)
    print(f"Precision-Recall curve data saved to '{pr_data_file}'")
   
    
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python apply_model.py <input_file> <model_filename> <output_tag>")
        sys.exit(1)

    input_file = sys.argv[1]
    model_filename = sys.argv[2]
    output_tag = sys.argv[3]
    apply_model_to_blind_data(input_file, model_filename, output_tag)


