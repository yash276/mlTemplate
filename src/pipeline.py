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
    # Step 1 Perform The Automatic EDA
    input_cfg = cfg['input']
    train_df = pd.read_csv(input_cfg['train_file'])
    eda.eda(dataframe = train_df)
    # Step 2 Perform Cross Validation
    cv_cfg = cfg['cross_validation']
    cv = cross_validation.CrossValidation(dataframe = train_df,
                                          cv_cfg = cv_cfg)
    train_df = cv.split()
    # Step 3 Seperate Categorical and Numerical Features
    # Perform specific Feature Engineering and Selection
    cat_feats_cfg = cfg['catergorical_features']
    cat_feats_cfg['catergorical_features'] = 
    
    cat_feats = categorical_features.CategoricalFeatures(
        dataframe=train_df,
        cat_feats_cfg=cat_feats_cfg 
    )
    
    output_df = cat_feats.fit_transform()
    print(output_df.head())
    # Step 4 Combine Back The Categorical and Numerical Features
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