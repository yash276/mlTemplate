
export TRAINING_DATA=input/trainFolds.csv
export TEST_DATA=input/test_cat.csv

export MODEL=$1

export FOLD=0
python3 -m src.train

