import yaml
import os
import pandas as pd
from pandas_profiling import ProfileReport
from yaml.loader import SafeLoader

def eda(cfg : dict) -> None :    
    """
    The function generates an automated EDA report and save it in a HTML format.
    The generated report can be shared across and is platform independent.

    Args:
        cfg (dict): input dictionary which should have the following format
                    cfg = {
                            'path': "input/train.csv" 
                        }    
    """
    df = pd.read_csv(cfg['path'])
    profile = ProfileReport(df,title="Framingham Kaggle Data EDA Report")
    profile.to_file("EDA_PandasProfling_3.html")

if __name__ == "__main__":
    CONFIG = os.environ.get("CONFIG")
    print(CONFIG)
    with open(CONFIG) as f:
        cfg = yaml.load(f,Loader=SafeLoader)
    
    input = cfg['input']
    eda(input)