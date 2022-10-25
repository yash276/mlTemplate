from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
import os 

from sklearn import metrics
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():

    predictions = None
    
    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        test_idx = df['id'].values
        train_columns = joblib.load(os.path.join("model",f"{MODEL}_{FOLD}_columns.pkl"))
        label_encoders = joblib.load(os.path.join("model",f"{MODEL}_{FOLD}_label_encoder.pkl"))    
        clf = joblib.load(os.path.join("model",f"{MODEL}_{FOLD}.pkl"))
        for column in train_columns:
            lbl = label_encoders[column]
            df.loc[:,column] = lbl.transform(df[column].values.tolist())
        
        df = df[train_columns]
        preds = clf.predict_proba(df)[:,1]
        
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5 
    print(predictions)
    
if __name__ == "__main__":
    predict()