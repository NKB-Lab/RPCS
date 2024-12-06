import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def train_and_save_models(input_file):
    # Load the dataset
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        sys.exit(1)

    # Replace '.' with NaN in the data
    data.replace('.', np.nan, inplace=True)
    
    # Extracting features and labels
    feature_columns = data.columns[1:-1]  # Skip first column (ID) and last column (label)
    X = data[feature_columns]
    y = data[data.columns[-1]].map({"Yes": 1, "No": 0})  # Convert labels to binary

    # Handle missing values using MICE
    imputer = IterativeImputer(random_state=0, max_iter=10)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=0)

    # Initialize models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=0),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=0),
        "SVM": SVC(kernel='linear', probability=True, random_state=0),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=0)
    }


    # Create output directory
    output_dir = "trained_models"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(output_dir, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"{name} model saved to {model_path}")

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
        auc_pr = average_precision_score(y_test, y_pred_prob) if y_pred_prob is not None else None

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC-ROC": auc_roc,
            "AUC-PR": auc_pr
        })

    # Save evaluation metrics
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "evaluation_metrics.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation metrics saved to {results_path}")
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_and_save_model.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    train_and_save_models(input_file)
