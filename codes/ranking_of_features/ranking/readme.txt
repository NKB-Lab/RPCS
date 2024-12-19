
## Processing input data

cat path_to/driver_dataset/driver.txt > Input_Iteration1.txt
cat path_to/non-driver_dataset/Iteration1.txt | sed 1d >> Input_Iteration1.txt

## applying RFE
python3 ranking.py Input_Iteration1.txt Output_Iteration1.csv

## Need to repeat the above steps for non-driver_dataset/Iteration{1..100}.txt
