# import files/functions from our code
from . import eda
# import python libraries
import os
import yaml
from yaml.loader import SafeLoader

def pipeline(cfg : dict):
    # Step 1 Perform an Automated EDA and store the results 
    input = cfg['input']
    eda.eda(input)
    # Step 2 Perform Cross Validation
    # Step 3

if __name__ == "__main__":
    # read/create the config dict for pipeline
    CONFIG = os.environ.get("CONFIG")
    with open(CONFIG) as f:
        cfg = yaml.load(f,Loader=SafeLoader)
    # send the config dict to the pipeline function
    pipeline(cfg=cfg)