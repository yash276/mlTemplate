# * DONE : Create Automated EDA using SweetViz or Pandas Profiling --- Automate Script
# cross validation --- Automate Script
# seperate numerical and categorical features --- Manual. Can we automate this somehow??
    # TODO: dtype
# Categorcial Features
    # Handle NAN Values -- automate script
    # One Hot Encoding --- automate script
    # Need to handle Rare and NAN Values.
# Numerical Features
    # 
# Modelling --- Start with Linear and Logistic Regression
# All the test Anova , T-Test, Chi-Squared
# Get insights about Feature Selection and Engineering by looking at things like 
# p-values and violation of Regression
# Perform Problem Specific Data Manipulation. 
# TODO: PCA
# TODO: Recommendation System to tell what to do with what features.
# TODO: Tool as an IP to Business 
export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv
export CONFIG=config.yaml
export MODEL=$1

python3 -m src.pipeline
# FOLD=0 python3 -m src.train
# FOLD=1 python3 -m src.train
# FOLD=2 python3 -m src.train
# FOLD=3 python3 -m src.train
# FOLD=4 python3 -m src.train
#python3 -m src.predict