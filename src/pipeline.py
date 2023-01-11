# import files/functions from our code
from . import eda
from . import categorical_features
from . import cross_validation
from . import train
from . import regression_diagnosis
# import python libraries
import os
import yaml
import pandas as pd
from yaml.loader import SafeLoader

def pipeline(cfg : dict):
    # read train and test files
    input_cfg = cfg['input']
    train_df = pd.read_csv(input_cfg['train_file'])
    test_df = pd.read_csv(input_cfg['test_file'])
    # create deep copies of dataframes so that we can use original whereever required
    train_df_d_copy = train_df.copy(deep=True)
    test_df_d_copy = test_df.copy(deep=True)
    
    test_df_d_copy[input_cfg['target_cols']] = -1
    train_len = len(train_df_d_copy)
    
    # Step 1 Perform The Automatic EDA
    eda.eda(input_cfg = input_cfg)
    # Step 2 Seperate Categorical Features for Train and Test
    cat_feats_cfg = cfg['categorical_features']
    if "cols" in cat_feats_cfg:
        cat_feats = cat_feats_cfg['cols']
    else:
        # select all the columns that have dtype as object as categorical and remaning as numerical
        # convert them to list
        train_cats = train_df_d_copy.select_dtypes(include='object').columns.tolist()
        test_cats = test_df_d_copy.select_dtypes(include='object').columns.tolist()
        
        train_nums = train_df_d_copy.select_dtypes(exclude='object').columns.tolist()
        test_nums = test_df_d_copy.select_dtypes(exclude='object').columns.tolist()
        # to remove dulicates from train and test data
        cat_feats = list(set(train_cats + test_cats))
        cat_feats_cfg['cols'] = cat_feats
        num_feats = list(set(train_nums + test_nums))
        
    full_dataframe = pd.concat([train_df_d_copy, test_df_d_copy]).reset_index(drop=True)
    cat_feats = categorical_features.CategoricalFeatures(
        dataframe = full_dataframe,
        cat_feats_cfg = cat_feats_cfg 
    )
    full_df_cats = cat_feats.fit_transform()
    
    #split the training and test data again 
    train_df_d_copy = full_df_cats.iloc[:train_len , :]
    test_df_d_copy = full_df_cats.iloc[train_len: , :]
    # Step 3 perform numerical feature engineering
    # Step 4 Perform Cross Validation
    cv_cfg = cfg['cross_validation']
    cv_cfg['target_cols'] = input_cfg['target_cols']
    cv = cross_validation.CrossValidation(dataframe = train_df_d_copy,
                                          cv_cfg = cv_cfg)
    train_df_d_copy = cv.split()
    # Step 5 Model Dispatcher and Training
    # Fill in the train config with the details of above steps 
    train_cfg = cfg['training']
    train_cfg['target_cols'] = input_cfg['target_cols']
    train_cfg['output_path'] = input_cfg['output_path']
    clfs = []
    # run the training for each fold and save the model for each fold
    for fold in range(cv_cfg['num_folds']):
        train_cfg['num_folds'] = fold
        clfs.append(train.train(dataframe= train_df_d_copy , train_cfg=train_cfg))
    
    # Step 6 Regression Analysis
    diagnosis_cfg = cfg['diagnosis']
    diagnosis_cfg['classifiers'] = clfs
    diagnosis_cfg['cat_cols'] = cat_feats_cfg['cols']
    diagnosis_cfg['target_cols'] = input_cfg['target_cols']
    diagnosis_cfg['output_path'] = input_cfg['output_path']
    # !Note Send original Train and Test Dataframes for Diagnosis of Regression
    regression_diagnosis.RegressionDiagnosis(train_df,test_df,diagnosis_cfg)
    # checking the goddness of fit
    # And checking the statistical significance
    # Step 7 Prediction

if __name__ == "__main__":
    # read/create the config dict for pipeline
    CONFIG = os.environ.get("CONFIG")
    with open(CONFIG) as f:
        cfg = yaml.load(f,Loader=SafeLoader)
    # send the config dict to the pipeline function
    pipeline(cfg=cfg)