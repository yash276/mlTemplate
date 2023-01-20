
from . import feature_selection
import joblib
import pandas as pd

def predict(
            dataframe: pd.DataFrame,
            train_cfg: dict):

    predictions = None
    feature_selection_cfg = train_cfg['feature_selection']
    
    feature_select = feature_selection.FeatureSelection(
        train_df= dataframe,
        feature_selection_cfg= feature_selection_cfg,
        train=False
    )
    dataframe_feats = feature_select.get_df()
    
    for index, clfs_path in enumerate(train_cfg['training']['clfs_path']):  
        clf = joblib.load(clfs_path)
        preds = clf.predict_proba(dataframe_feats.values)[:,1]
        
        if index == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= len(train_cfg['training']['clfs_path'])
    
if __name__ == "__main__":
    predict()