# import files/functions from our code
from . import eda
from . import feature_selection
from . import cross_validation
from . import train
from . import predict

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
    
    # Step 1 Perform The Automatic EDA
    eda.eda(input_cfg = input_cfg)
    
    # Step 2 Perform Feature Selection for Categorical and Numerical Features
    feature_selection_cfg = cfg['feature_selection']
    feature_selection_cfg['target_cols'] = input_cfg['target_cols']
    feature_selection_cfg['output_path'] = input_cfg['output_path']
    
    feature_select = feature_selection.FeatureSelection(
        train_df= train_df_d_copy,
        test_df= test_df_d_copy,
        feature_selection_cfg= feature_selection_cfg
    )
    train_df_d_copy,  test_df_d_copy = feature_select.get_df()
    
    # Step 3 Perform Cross Validation
    cv_cfg = cfg['cross_validation']
    cv_cfg['target_cols'] = input_cfg['target_cols']
    cv = cross_validation.CrossValidation(dataframe = train_df_d_copy,
                                          cv_cfg = cv_cfg)
    train_df_d_copy = cv.split()
    cfg['cross_validation'] = cv_cfg
    
    # Step 4 Model Dispatcher and Training
    # Fill in the train config with the details of above steps 
    train_cfg = cfg['training']
    train_cfg['target_cols'] = input_cfg['target_cols']
    train_cfg['output_path'] = input_cfg['output_path']
    train_cfg['clfs'] = []
    # run the training for each fold and save the model for each fold
    for fold in range(cv_cfg['num_folds']):
        train_cfg['num_folds'] = fold
        train_cfg['clfs'].append(train.train(dataframe= train_df_d_copy , train_cfg=train_cfg))
    cfg['training'] = train_cfg
    # Step 5 Prediction
    predict_cfg = cfg['predict'] = {}

    predict_cfg['clfs'] = clfs
    predict_cfg['num_folds'] = cv_cfg['num_folds']
    predict_cfg['encoders'] = 
    predict_cfg['output_path'] = input_cfg['output_path']
    
    predict.predict(train_df, test_df , predict_cfg)
    
if __name__ == "__main__":
    # read/create the config dict for pipeline
    CONFIG = os.environ.get("CONFIG")
    with open(CONFIG) as f:
        cfg = yaml.load(f,Loader=SafeLoader)
    # send the config dict to the pipeline function
    pipeline(cfg=cfg)