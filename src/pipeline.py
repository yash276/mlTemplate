# import files/functions from our code
from . import eda
from . import cross_validation
from . import categorical_features
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
    test_df[input_cfg['target_cols']] = -1
    train_len = len(train_df)
    test_len = len(test_df)
    # Step 1 Perform The Automatic EDA
    eda.eda(input_cfg = input_cfg)
    # Step 2 Seperate Categorical Features for Train and Test
    cat_feats_cfg = cfg['categorical_features']
    if "cols" in cat_feats_cfg:
        cat_feats = cat_feats_cfg['cols']
    else:
        # select all the columns that have dtype as object as categorical and remaning as numerical
        # convert them to list
        train_cats = train_df.select_dtypes(include='object').columns.tolist()
        test_cats = test_df.select_dtypes(include='object').columns.tolist()
        
        train_nums = train_df.select_dtypes(exclude='object').columns.tolist()
        test_nums = test_df.select_dtypes(exclude='object').columns.tolist()
        # to remove dulicates from train and test data
        cat_feats = list(set(train_cats + test_cats))
        num_feats = list(set(train_nums + test_nums))
        
    cat_feats_cfg['cols'] = cat_feats
    full_dataframe = pd.concat([train_df, test_df]).reset_index(drop=True)
    cat_feats = categorical_features.CategoricalFeatures(
        dataframe = full_dataframe,
        cat_feats_cfg = cat_feats_cfg 
    )
    full_df_cats = cat_feats.fit_transform()
    
    #split the training and test data again 
    train_df = full_df_cats.iloc[:train_len , :]
    test_df = full_df_cats.iloc[train_len: , :]
    # Step 3 perform numerical feature engineering
    # Step 4 Perform Cross Validation
    cv_cfg = cfg['cross_validation']
    cv = cross_validation.CrossValidation(dataframe = train_df,
                                          cv_cfg = cv_cfg)
    train_df = cv.split()
    # Step 5 Train the Model
    # Step 6 Check The Underlying Assumptions and Gain Insights About The Data
    # Step 7 Prediction

if __name__ == "__main__":
    # read/create the config dict for pipeline
    CONFIG = os.environ.get("CONFIG")
    with open(CONFIG) as f:
        cfg = yaml.load(f,Loader=SafeLoader)
    # send the config dict to the pipeline function
    pipeline(cfg=cfg)