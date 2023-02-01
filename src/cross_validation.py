import pandas as pd
from sklearn import model_selection
class CrossValidation:
    """
    This class is responsible for performing Cross-Validation of the Training Data 
    based on the input provided in the Config
    """
    def __init__(self,
                 dataframe: pd.DataFrame, 
                 cv_cfg: dict
                 ):
        """

        Args:
            dataframe (pd.DataFrame): Input DataFrame on which the cross validation is to be performed
            
            cv_cfg (dict): The dictinoary should have the following fromat
            KEEP THE KEY VALUES AS GIVEN BELOW!!!
            
            cross_validation: {
                problem_type(string): supported
                "binary_classification",
                "multiclass_classification",
                "single_column_regression",
                "multi_column_regression",
                "multilabel_classification"
                multilabel_delimiter(string, optional): the character that seperates your multilabel in the input data,
                shuffle(bool, optional): if you want to shuffle the input data,
                num_folds(int, optional): num of folds you want yo split the input data,
                random_state(int, optional): random state you want to shuffle the data
                }
        """
        self.dataframe = dataframe
        self.cv_cfg = cv_cfg
        self.target_cols = cv_cfg['target_cols']
        self.num_targets = len(self.target_cols)
        self.problem_type = cv_cfg['problem_type']
        # assiging some default values if the keys are missing
        if 'multilabel_delimiter' in cv_cfg:
            self.multilabel_delimiter = cv_cfg['multilabel_delimiter']
        else:
            self.multilabel_delimiter = " "
            
        if 'num_folds' in cv_cfg:
            self.num_folds = cv_cfg['num_folds']
        else:
            self.num_folds = 5
            
        if 'shuffle' in cv_cfg:
            self.shuffle = cv_cfg['shuffle']
        else:
            self.shuffle = True
            
        if 'random_state' in cv_cfg:
            self.random_state = cv_cfg['random_state']
        else:
            self.random_state = 42
            
        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        self.dataframe['kfold'] = -1
    
    def split(self) -> pd.DataFrame:
        """
        Performs cross-validation on the training dataframe based on problem statement defined in config.

        Raises:
            Exception: Invalid number of target for binary_classification and 
            multiclass_classification problem type
            
            Exception: Only one unique value found in Target for binary_classification and 
            multiclass_classification problem type
            
            Exception: Invalid number of target for single_column_regression and 
            multi_column_regression problem type
            
            Exception: Invalid number of target for multilabel_classification problem type
            
            Exception: Funnny problem type not Understood

        Returns:
            pd.DataFrame: shuffled dataframe along with the folds information in kfold column
        """
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets > 1 :
                raise Exception("Invalid number of target for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                # single target value so no point in creating the model
                raise Exception("Only one unique value found for Target")
            elif unique_values > 1:
                # use stratified k-fold
                kf =  model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                      shuffle=False
                                                      )
                for fold,(train_idx,val_idx) in enumerate(kf.split(X=self.dataframe,y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx,'kfold'] = fold
            
        elif self.problem_type in ("single_column_regression","multi_column_regression"):
            if self.num_targets != 1 and self.problem_type == "single_column_regression" :
                raise Exception("Invalid number of target for this problem type")
            if self.num_targets < 1 and self.problem_type == "multi_column_regression" :
                raise Exception("Invalid number of target for this problem type")
            
            kf = model_selection.KFold(n_splits=self.num_folds,
                                        shuffle=False
                                        )
            for fold,(train_idx,val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx,'kfold'] = fold

        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split('_')[1])
            num_holdout_samples = int(len(self.dataframe) * (holdout_percentage) / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples,"kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:,"kfold"] = 1
        
        elif self.problem_type == "multilabel_classification":
            if self.num_targets !=1 :
                    raise Exception("Invalid number of target for this problem type")
            target = self.dataframe[self.target_cols[0]].apply(lambda x : len(str(x).split(self.multilabel_delimiter)))
            kf =  model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold,(train_idx,val_idx) in enumerate(kf.split(X=self.dataframe,y=target)):
                self.dataframe.loc[val_idx,'kfold'] = fold
        else:
            raise Exception("Funnny problem type not Understood")
            
        return self.dataframe
    
    def get_config(self):
        """
        It returns the current input config that the initialized object is operating on.
        
        Returns:
            dict: the enitre feature selection config
        """
        return self.cv_cfg