from . import categorical_features
from . import numerical_features
import pandas as pd

class FeatureSelection:
    def __init__(self,
                 train_df: pd.DataFrame,
                 feature_selection_cfg : dict,
                 train: bool,
                 test_df=None
                 ) -> None:
        
        self.train = train
        # extract the datasets and create a full dataset
        self.train_df = train_df
        self.train_df = self.train_df.drop(feature_selection_cfg['cols_to_drop'],axis=1)
        
        # Drop the target column from train.
        temp_train_df = self.train_df
        
        if self.train:
            temp_train_df = self.train_df.drop(feature_selection_cfg['target_cols'],axis=1)
            self.test_df = test_df
            self.test_df = self.test_df.drop(feature_selection_cfg['cols_to_drop'],axis=1)
            # get the length of the training data
            # it will be usefull to seperate the training and test data later on 
            self.train_len = len(self.train_df)
        
        # extract categorical and numerical features config
        self.feature_select_cfg = feature_selection_cfg
        self.cat_feats_cfg = feature_selection_cfg['categorical_features']
        self.cat_feats_cfg['target_cols'] = feature_selection_cfg['target_cols']
        self.cat_feats_cfg['output_path'] = feature_selection_cfg['output_path']
        
        self.num_feats_cfg = feature_selection_cfg['numerical_features']
        self.num_feats_cfg['target_cols'] = feature_selection_cfg['target_cols']
        self.num_feats_cfg['output_path'] = feature_selection_cfg['output_path']
        
        # check if categorical and numerical features are given in the config
        # if not figure them out automatically
        if "cols" not in self.cat_feats_cfg:
            # select categorical features with low cardinality
            self.cat_feats_cfg['cols'] = [cname for cname in temp_train_df.columns if 
                                    temp_train_df[cname].nunique() < 10 and
                                    temp_train_df[cname].dtype == "object"]
        if 'cols' not in self.num_feats_cfg:
            # select numerical features whose dtype is int or float
            self.num_feats_cfg['cols'] = [cname for cname in temp_train_df.columns if 
                                    temp_train_df[cname].dtype in ['int64', 'float64']]
        
        if self.train:
            # after getting both the numerical and catergorical features
            # create an concatenated data from train and test data
            # drop all the other columns not present in categorical and numerical lists
            self.full_dataframe = pd.concat([temp_train_df, self.test_df]).reset_index(drop=True)
        else:
            self.full_dataframe = temp_train_df
            
        for feats in temp_train_df:
            if feats not in self.cat_feats_cfg['cols'] and feats not in self.num_feats_cfg['cols']:
                self.full_dataframe = self.full_dataframe.drop(feats,axis=1)
        
        # create the categorical class constructor
        self.cat_feats = categorical_features.CategoricalFeatures(
            dataframe = self.full_dataframe[self.cat_feats_cfg['cols']],
            cat_feats_cfg = self.cat_feats_cfg,
            train = self.train
        )
        # create the numnerical class constructor
        self.num_feats = numerical_features.NumericalFeatures(
            dataframe= self.full_dataframe[self.num_feats_cfg['cols']],
            num_feats_cfg= self.num_feats_cfg,
            train = self.train
        )
        
        if self.train:
                # Run run_tests and select_best according to config
            if feature_selection_cfg['run_tests']:
                self.run_tests()
            if feature_selection_cfg['select_best']:
                self.select_best()
    
    def run_tests(self):
        # perform and produce results for some tests
        self.cat_feats.chi2_test(self.train_df)
    
    def select_best(self):
        # select the best features 
        self.feature_select_cfg['categorical_features']['cols'] = self.cat_feats_cfg['cols'] = self.cat_feats.select_best(self.train_df)
    
    def get_df(self):
        # beforing sending the dataframe make sure to perform transformation on the data
        # to make it ready for the training
        full_cats_df = self.cat_feats.fit_transform()
        full_num_df = self.num_feats.fit_transform()
        
        if self.train:
            # get the training data from both catergorical and numerical dataframe
            # combine them to create a single training dataframe
            train_cats_df = full_cats_df.iloc[:self.train_len , :]
            train_num_df = full_num_df.iloc[:self.train_len , :]
            train_df = pd.concat([train_cats_df,train_num_df],axis=1)
            
            # Perform the same operation for the test data
            test_cats_df = full_cats_df.iloc[self.train_len: , :]
            test_num_df = full_num_df.iloc[self.train_len: , :]
            test_df = pd.concat([test_cats_df,test_num_df],axis=1)
            
            # add the target column back to the train df
            train_df[self.cat_feats_cfg['target_cols']] = self.train_df[self.cat_feats_cfg['target_cols']].values.tolist()
            
            return train_df , test_df
        else:
            train_df = pd.concat([full_cats_df,full_num_df],axis=1)
            return train_df
    
    def get_config(self):
        self.feature_select_cfg['categorical_features'] = self.cat_feats_cfg = self.cat_feats.get_config()
        return self.feature_select_cfg