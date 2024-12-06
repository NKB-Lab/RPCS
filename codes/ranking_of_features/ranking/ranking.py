import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Feature ranking with RFE using RF, LR, and SVM")
parser.add_argument("input_file", type=str, help="Path to the input CSV file")
parser.add_argument("output_file", type=str, help="Path to save the output CSV file")
args = parser.parse_args()

# Load the data
df = pd.read_csv(args.input_file, delimiter='\t')

# Separate identifiers, features, and labels
identifiers = df.iloc[:, 0]
features = df.iloc[:, 1:-1].replace('.', np.nan).astype(float)
labels = df.iloc[:, -1]

# Step 1: Impute missing values using MICE (IterativeImputer in Scikit-Learn)
imputer = IterativeImputer(max_iter=10, random_state=0)
features_imputed = imputer.fit_transform(features)
features_df = pd.DataFrame(features_imputed, columns=features.columns)

# Step 2: Initialize the models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=0),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=0),
    'SVM': SVC(kernel='linear', random_state=0)
}

# Step 3: Perform Recursive Feature Elimination (RFE) for each model
feature_rankings = {}

for model_name, model in models.items():
    rfe = RFE(estimator=model, n_features_to_select=1, step=1)
    rfe.fit(features_df, labels)
    feature_rankings[model_name] = rfe.ranking_

# Step 4: Compile rankings into a DataFrame
ranking_df = pd.DataFrame(feature_rankings, index=features.columns)
ranking_df.columns = ['Rank_' + col for col in ranking_df.columns]

# Step 5: Save the rankings to a CSV file
ranking_df.to_csv(args.output_file, index=True)

print(f"Ranking results saved to {args.output_file}")
