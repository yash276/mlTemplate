import yaml
import os
import pandas as pd
from pandas_profiling import ProfileReport
from yaml.loader import SafeLoader

def eda(input_cfg: dict) -> None :    
    """
    The function generates an automated EDA report and save it in a HTML format.
    The generated report can be shared across and is platform independent.
    
    Args:
        input_cfg (dict): input dictionary which should have the following format
            input_cfg = {
                train_file: "input/train.csv",
                test_file: "",
                validation_file: "",
                output_path: "output"
            }
    """
    dataframe = pd.read_csv(input_cfg['train_file'])
    output_file_path = os.path.join(input_cfg['output_path'],'EDA.html')
    
    profile = ProfileReport(dataframe,title="EDA Report")
    profile.to_file(output_file_path)

if __name__ == "__main__":
    CONFIG = os.environ.get("CONFIG")
    with open(CONFIG) as f:
        cfg = yaml.load(f,Loader=SafeLoader)
    
    input_cfg = cfg['input']
    train_df = pd.read_csv(input_cfg['train_file'])
    eda.eda(dataframe = train_df)