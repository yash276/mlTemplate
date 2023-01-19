from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
import os 

from . import dispatcher
from . import feature_selection
import joblib
from sklearn import metrics

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

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
    print(dataframe_feats)
    # for FOLD in range(5):
    #     df = pd.read_csv(TEST_DATA)
    #     test_idx = df['id'].values
    #     train_columns = joblib.load(os.path.join("model",f"{MODEL}_{FOLD}_columns.pkl"))
    #     label_encoders = joblib.load(os.path.join("model",f"{MODEL}_{FOLD}_label_encoder.pkl"))    
    #     clf = joblib.load(os.path.join("model",f"{MODEL}_{FOLD}.pkl"))
    #     for column in train_columns:
    #         lbl = label_encoders[column]
    #         df.loc[:,column] = lbl.transform(df[column].values.tolist())
        
    #     df = df[train_columns]
    #     preds = clf.predict_proba(df)[:,1]
        
    #     if FOLD == 0:
    #         predictions = preds
    #     else:
    #         predictions += preds
    
    # predictions /= 5 
    # print(predictions)
    
if __name__ == "__main__":
    predict()