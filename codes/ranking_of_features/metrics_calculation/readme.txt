
## Processing input data

cat path_to/driver_dataset/driver.txt | sed 's|\t|,|g' > Input_Iteration1.csv
cat path_to/non-driver_dataset/Iteration1.txt | sed 1d | sed 's|\t|,|g' >> Input_Iteration1.csv

## calculating metrics

python3 metrics.py --input_file Input_Iteration1.csv --output_tag Iteration1_output

## Need to repeat the above steps for non-driver_dataset/Iteration{1..100}.txt
