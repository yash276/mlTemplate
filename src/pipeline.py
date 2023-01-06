# import files/functions from our code
from . import eda
from . import cross_validation
# import python libraries
import os
import yaml
from yaml.loader import SafeLoader

def pipeline(cfg : dict):
    # Step 1 Perform The Automatic EDA
    input_cfg = cfg['input']
    eda.eda(input_cfg = input_cfg)
    # Step 2 Perform Cross Validation
    cv_cfg = cfg['cross_validation']
    cv = cross_validation.CrossValidation(input_cfg = input_cfg,
                                          cv_cfg = cv_cfg)
    train_df = cv.split()
    print(train_df.head())
    # Step 3 Seperate Categorical and Numerical Features
    # Perform specific Feature Engineering and Selection
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