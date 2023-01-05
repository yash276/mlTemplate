# import files/functions from our code
from . import eda
from . import cross_validation
# import python libraries
import os
import yaml
import pandas as pd
from yaml.loader import SafeLoader

def pipeline(cfg : dict):
    # Step 1 Perform the automatic EDA
    input_cfg = cfg['input']
    eda.eda(input_cfg = input_cfg)
    # Step 3 Perform Cross Validation
    cv_cfg = cfg['cross_validation']
    cv = cross_validation.CrossValidation(input_cfg =input_cfg,
                                          cv_cfg = cv_cfg)
    df_split = cv.split()
    print(df_split.head()) 
    print(df_split.kfold.value_counts())
    # Step 3

if __name__ == "__main__":
    # read/create the config dict for pipeline
    CONFIG = os.environ.get("CONFIG")
    with open(CONFIG) as f:
        cfg = yaml.load(f,Loader=SafeLoader)
    # send the config dict to the pipeline function
    pipeline(cfg=cfg)