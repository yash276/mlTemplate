from . import categorical_features
from . import numerical_features
import pandas as pd

class FeatureSelection:
    def __init__(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 feature_selection_cfg : dict
                 ) -> None:
        # extract the datasets and create a full dataset
        self.train_df = train_df
        self.test_df = test_df
        # get the length of the training data
        # it will be usefull to seperate the training and test data later on 
        self.train_len = len(self.train_df)
        # Drop the target column from train.
        temp_train_df = self.train_df.drop(feature_selection_cfg['target_cols'],axis=1)
        
        # extract categorical and numerical features config
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
        
        
        # after getting both the numerical and catergorical features
        # create an concatenated data from train and test data
        # drop all the other columns not present in categorical and numerical lists
        self.full_dataframe = pd.concat([temp_train_df, self.test_df]).reset_index(drop=True)
        for feats in temp_train_df:
            if feats not in self.cat_feats_cfg['cols'] and feats not in self.num_feats_cfg['cols']:
                self.full_dataframe = self.full_dataframe.drop(feats,axis=1)
        print(self.full_dataframe)
        print("------------------------------------------------------")
        # create the categorical class constructor
        self.cat_feats = categorical_features.CategoricalFeatures(
            dataframe = self.full_dataframe[self.cat_feats_cfg['cols']],
            cat_feats_cfg = self.cat_feats_cfg 
        )
        # create the numnerical class constructor
        self.num_feats = numerical_features.NumericalFeatures(
            dataframe= self.full_dataframe[self.num_feats_cfg['cols']],
            num_feats_cfg= self.num_feats_cfg
        )
    
    def run_tests(self):
        # perform and produce results for some tests
        self.cat_feats.chi2_test(self.train_df)
    
    def select_best(self):
        # select the best features 
        self.cat_feats.select_best(self.train_df)
    
    # def fit_transform(self):
    #     # then transform the features  
    #     self.full_dataframe = self.cat_feats.fit_transform()

    
    def get_df(self):
        # beforing sending the dataframe make sure to perform transformation on the data
        # to make it ready for the training
        full_cats_df = self.cat_feats.fit_transform()
        full_num_df = self.num_feats.fit_transform()
        
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
        train_df[self.cat_feats_cfg['target_cols']] = self.train_df[self.cat_feats_cfg['target_cols']]
        
        return train_df , test_df