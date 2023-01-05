from sklearn import model_selection
import pandas as pd
class CrossValidation:
    """
    binary classification
    multi class classification
    multi label classification
    single column regression
    multi column regression
    holdout
    """
    def __init__(self,
                 input_cfg: dict, 
                 cv_cfg: dict
                 ) -> pd.DataFrame:
        
        self.dataframe = pd.read_csv(input_cfg['train_file'])
        self.target_cols = cv_cfg['target_cols']
        self.num_targets = len(self.target_cols)
        self.problem_type = cv_cfg['problem_type']
        self.multilabel_delimiter = cv_cfg['multilabel_delimiter']
        self.num_folds = cv_cfg['num_folds']
        self.shuffle = cv_cfg['shuffle']
        self.random_state = cv_cfg['random_state']
        
        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        self.dataframe['kfold'] = -1
    
    def split(self):
        """_summary_

        Raises:
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_

        Returns:
            _type_: _description_
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
                    print(len(train_idx),len(val_idx))
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
                print(len(train_idx),len(val_idx))
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
                print(len(train_idx),len(val_idx))
                self.dataframe.loc[val_idx,'kfold'] = fold
        else:
            raise Exception("Funnny problem type not Understood")
            
        return self.dataframe