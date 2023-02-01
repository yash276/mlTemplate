from . import dispatcher
import joblib
import pandas as pd
from sklearn import metrics

def train(
    dataframe: pd.DataFrame,
    train_cfg: dict
    )-> str:
    """
    The functions takes in Training DataFrame and Train Config.
    Split the DataFrame in Train and Validation sets.
    Trains a model according to config and store it in the output path. 

    Args:
        dataframe (pd.DataFrame): Training DataFrame
        train_cfg (dict): The dictinoary should have the following fromat
            KEEP THE KEY VALUES AS GIVEN BELOW!!!
            
            training: {
                model: 
                "logistic" for Logistic Regression
                "linear" for Liner Regression
            }

    Returns:
        str: the path to the trained classifier.
    """
    fold = train_cfg['num_folds']
    # get the Training and validation data for this fold
    # training data is where the kfold is not equal to the fold
    # validation data is where the kfold is equal to the fold
    train_df = dataframe[dataframe.kfold != fold].reset_index(drop=True)
    val_df = dataframe[dataframe.kfold==fold].reset_index(drop=True)
    
    # drop the kfold and target column    
    # convert it into a numpy array
    x_train = train_df.drop(['kfold'] + train_cfg['target_cols'],axis=1).values
    y_train = train_df[train_cfg['target_cols']].values
    # perform the same for validation
    x_val = val_df.drop(['kfold'] + train_cfg['target_cols'],axis=1).values
    y_val = val_df[train_cfg['target_cols']].values
    
    # fetch the model from the model dispatcher
    clf = dispatcher.models[train_cfg['model']]
    
    #fit the model on the training data
    clf.fit(x_train,y_train)
    
    # create probabilities for validation samples
    preds = clf.predict_proba(x_val)[:,1]

    # get roc auc score
    auc = metrics.roc_auc_score(y_val,preds)
    # print the auc score
    print(f"Fold={fold}, AUC SCORE={auc}") 
    # save the model along with fold number
    clf_path = f"{train_cfg['output_path']}/{train_cfg['model']}_{train_cfg['num_folds']}.pkl"
    joblib.dump(clf,clf_path)
    
    return clf_path