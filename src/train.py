from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
import os 

from sklearn import metrics
from . import dispatcher
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
    trainDF = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    validDF = df[df.kfold==FOLD]

    ytrain = trainDF.target.values
    yvalid = validDF.target.values

    trainDF = trainDF.drop(["id","target","kfold"],axis=1)
    validDF = validDF.drop(["id","target","kfold"],axis=1)
    
    validDF = validDF[trainDF.columns]

    labelEncoders = []

    for column in trainDF.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(trainDF[column].values.tolist() + validDF[column].values.tolist())
        trainDF.loc[:,column] = lbl.transform(trainDF[column].values.tolist())
        validDF.loc[:,column] = lbl.transform(validDF[column].values.tolist())
        labelEncoders.append((column,lbl))
    
    # data ready to train
    classifier = dispatcher.models[MODEL]
    classifier.fit(trainDF,ytrain)
    preds = classifier.predict_proba(validDF)[:,1]
    # Roc_auc_score because data is skewed
    print(metrics.roc_auc_score(yvalid,preds))