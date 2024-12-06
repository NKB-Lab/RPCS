
python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv ./trained_models/LogisticRegression.pkl BRCA-TCGA_validation_11PCSAs_LogisticRegression
python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv ./trained_models/RandomForest.pkl BRCA-TCGA_validation_11PCSAs_RandomForest
python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv ./trained_models/SVM.pkl BRCA-TCGA_validation_11PCSAs_SVM
python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv ./trained_models/XGBoost.pkl BRCA-TCGA_validation_11PCSAs_XGBoost

python3 apply_model.py 20_hotspots_mutations_with_11scores.csv ./trained_models/RandomForest.pkl 20_hotspots_mutations_RandomForest

