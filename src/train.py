from . import dispatcher
from . import utils
import pandas as pd
from sklearn import metrics

class Train:
    def __init__(self,
                 dataframe: pd.DataFrame , 
                 train_cfg: dict) -> None:
        
        self.dataframe = dataframe
        self.train_cfg = train_cfg
        self.clf = []
        self.clf_path = []
        

    def train(self):
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
        first = True
        for fold in range(self.train_cfg['num_folds']):
            # get the Training and validation data for this fold
            # training data is where the kfold is not equal to the fold
            # validation data is where the kfold is equal to the fold
            train_df = self.dataframe[self.dataframe.kfold != fold].reset_index(drop=True)
            val_df = self.dataframe[self.dataframe.kfold==fold].reset_index(drop=True)
            
            # drop the kfold and target column    
            # convert it into a numpy array
            x_train = train_df.drop(['kfold'] + self.train_cfg['target_cols'],axis=1).values
            y_train = train_df[self.train_cfg['target_cols']].values
            # perform the same for validation
            x_val = val_df.drop(['kfold'] + self.train_cfg['target_cols'],axis=1).values
            y_val = val_df[self.train_cfg['target_cols']].values
            
            # fetch the model from the model dispatcher
            clf = dispatcher.models[self.train_cfg['model']]
            
            #fit the model on the training data
            clf.fit(x_train,y_train)
            
            # create probabilities for validation samples
            preds = clf.predict_proba(x_val)[:,1]
            # TODO: works only if you have single taget column
            # TODO: find a way to make it generic for n number of target columns
            residuals = y_val[:,0] - preds
            
            utils.scatter_plot(x_data=preds,
                               y_data=residuals,
                               title=f"Residuals_Vs_FittedValues_{fold}",
                               x_title="Predictions",
                               y_title="Residuals",
                               output_path=f"{self.train_cfg['output_path']}/Residuals_Vs_Fitted_Values_{fold}.html")
            
            # get roc auc score
            # auc = metrics.roc_auc_score(y_val,preds)
            # print("SCORE")
            # print the auc score
            # print(f"Fold={fold}, AUC SCORE={auc}") 
            # save the model along with fold number
            clf_path = f"{self.train_cfg['output_path']}/{self.train_cfg['model']}_{fold}.pkl"
            self.clf.append(clf)
            self.clf_path.append(clf_path)
        
        return self.clf, self.clf_path  
    
    def metrics(self):
        pass  