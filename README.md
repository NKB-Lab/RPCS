# RPCS
Ranking of Pathogenic and Conservation Scoring Algorithms (PCSAs) based on their ability to separating somatic pathogenic driver mutations from benign non-driver mutations in cancer.

Requirements:

Scikit-Learn - v1.2.2

Pandas - v1.4.3

Numpy - v1.22.2

## Processing input data

cat path_to/data/driver.txt > Input_"$line".txt
cat path_to/non-driver_dataset/Iteration1.txt | sed 1d >> Input_Iteration1.txt

## applying RFE
python3 ranking.py Input_Iteration1.txt Output_Iteration1.csv

## Need to repeat the above steps for non-driver_dataset/Iteration{1..100}.txt


# Train and save the model
python3 train_and_save_model.py HNSC-TCGA_training-data_11PCSAs.csv

# Applying model

python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv ./trained_models/LogisticRegression.pkl BRCA-TCGA_validation_11PCSAs_LogisticRegression
python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv ./trained_models/RandomForest.pkl BRCA-TCGA_validation_11PCSAs_RandomForest
python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv ./trained_models/SVM.pkl BRCA-TCGA_validation_11PCSAs_SVM
python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv ./trained_models/XGBoost.pkl BRCA-TCGA_validation_11PCSAs_XGBoost

python3 apply_model.py 20_hotspots_mutations_with_11scores.csv ./trained_models/RandomForest.pkl 20_hotspots_mutations_RandomForest

