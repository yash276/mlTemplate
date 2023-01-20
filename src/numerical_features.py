import pandas as pd

class NumericalFeatures:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_feats_cfg: dict,
        train: bool
        ) -> None:
        
        self.dataframe = dataframe
        self.num_feats = num_feats_cfg['cols']
        self.train = train
    
    def  select_best(self,dataframe: pd.DataFrame):
        pass
    
    
    def fit_transform(self):
        return self.dataframe[self.num_feats]
    