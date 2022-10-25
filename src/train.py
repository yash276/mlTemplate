from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
import os 

from sklearn import metrics
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING ={
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id","target","kfold"],axis=1)
    valid_df = valid_df.drop(["id","target","kfold"],axis=1)
    
    valid_df = valid_df[train_df.columns]

    label_encoders = []

    for column in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[column].values.tolist() + valid_df[column].values.tolist())
        train_df.loc[:,column] = lbl.transform(train_df[column].values.tolist())
        valid_df.loc[:,column] = lbl.transform(valid_df[column].values.tolist())
        label_encoders.append((column,lbl))
    
    # data ready to train
    classifier = dispatcher.models[MODEL]
    classifier.fit(train_df,ytrain)
    preds = classifier.predict_proba(valid_df)[:,1]
    # Roc_auc_score because data is skewed
    print(metrics.roc_auc_score(yvalid,preds))

    joblib.dump(label_encoders,f"model/{MODEL}LabelEncoder.pkl")
    joblib.dump(classifier,f"model/{MODEL}Classifier.pkl")