# RPCS
Ranking of Pathogenic and Conservation Scoring Algorithms (PCSAs) based on their ability to separating somatic pathogenic driver mutations from benign non-driver mutations in cancer.

Requirements:

Scikit-Learn - v1.2.2

Pandas - v1.4.3

Numpy - v1.22.2

# Processing input data

cat path_to/data/driver.txt > Input_Iteration1.txt

cat path_to/non-driver_dataset/Iteration1.txt | sed 1d >> Input_Iteration1.txt

# Applying RFE
python3 ranking.py Input_Iteration1.txt Output_Iteration1.csv

 Need to repeat the above steps for non-driver_dataset/Iteration{1..100}.txt


# Train and save the model
python3 train_and_save_model.py HNSC-TCGA_training-data_11PCSAs.csv

# Applying model on validation data

python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv codes/top-11_PCSAs/validation/trained_models/LogisticRegression.pkl BRCA-TCGA_validation_11PCSAs_LogisticRegression

python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv codes/top-11_PCSAs/validation/trained_models/RandomForest.pkl BRCA-TCGA_validation_11PCSAs_RandomForest

python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv codes/top-11_PCSAs/validation/trained_models/SVM.pkl BRCA-TCGA_validation_11PCSAs_SVM

python3 apply_model.py BRCA-TCGA_validation_11PCSAs.csv codes/top-11_PCSAs/validation/trained_models/XGBoost.pkl BRCA-TCGA_validation_11PCSAs_XGBoost

python3 apply_model.py 20_hotspots_mutations_with_11scores.csv codes/top-11_PCSAs/validation/trained_models/RandomForest.pkl 20_hotspots_mutations_RandomForest


# Applying model on new data

python3 run_models.py test.csv codes/top-11_PCSAs/models/imputer_model.pkl codes/top-11_PCSAs/models/RandomForest.pkl test_RF

python3 run_models.py test.csv codes/top-11_PCSAs/models/imputer_model.pkl codes/top-11_PCSAs/models/LogisticRegression.pkl test_LR

python3 run_models.py test.csv codes/top-11_PCSAs/models/imputer_model.pkl codes/top-11_PCSAs/models/SVM.pkl test_SVM

python3 run_models.py test.csv codes/top-11_PCSAs/models/imputer_model.pkl codes/top-11_PCSAs/models/XGBoost.pkl test_XGB

Input data structure:

Chromosome_Start_Ref_Alt,MutationAssessor_rankscore,PROVEAN_converted_rankscore,MetaLR_rankscore,M-CAP_rankscore,MutPred_rankscore,MVP_rankscore,DEOGEN2_rankscore,VARITY_R_rankscore,AlphaMissense_rankscore,fathmm-MKL_coding_rankscore,integrated_fitCons_rankscore,Status
19_13341002_C_T,0.53209,0.26200,0.98780,0.93624,0.79585,0.89338,0.76696,0.29084,0.74616,0.76055,0.61202,Yes
18_45374930_G_C,0.95212,0.95870,0.98763,0.93925,0.94965,0.95975,0.99935,0.95671,0.96936,0.95980,0.92422,Yes
6_26217383_G_C,.,0.75456,0.89007,0.74473,0.79585,0.99696,.,.,0.98268,0.75350,0.08003,No
20_34025155_C_G,.,0.42001,0.72506,0.52286,0.63707,0.51888,0.64413,.,0.65001,0.44369,0.14033,No
12_133233726_C_G,0.76847,0.24026,0.16073,0.15191,0.38176,0.40566,0.31355,0.42911,0.50817,0.84167,0.73137,Yes
12_121004691_G_C,0.12951,0.38151,0.85101,0.44001,0.51448,0.39508,0.28353,0.22822,0.08081,0.88512,0.73137,No
