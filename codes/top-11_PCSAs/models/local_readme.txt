
cat *TCGA_validation_11PCSAs.csv | grep "Chromosome_Start_Ref_Alt" -v | shuf | head -100 > test.csv

The above command was run in validation folder and then copy the test.csv to this folder
