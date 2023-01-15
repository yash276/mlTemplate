from . import utils
import os
import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import chi2  
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

class CategoricalFeatures:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cat_feats_cfg: dict
        ):
        # extract dataframe and the config values.
        self.dataframe = dataframe
        self.handle_na = cat_feats_cfg['handle_na']
        self.enc_types = cat_feats_cfg['enc_types']
        self.num_best = cat_feats_cfg['num_best']
        self.cat_feats = cat_feats_cfg['cols']
        self.target_cols = cat_feats_cfg['target_cols']
        self.output_path = cat_feats_cfg['output_path']
        # create empty dict's for storing encoders for features.
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None
        # hanlde NAN values if true.
        if self.handle_na:
            for feat in self.cat_feats:
                self.dataframe.loc[:,feat] = self.dataframe.loc[:,feat].astype(str).fillna("-9999999")
        self.dataframe_d_copy = self.dataframe.copy(deep=True)
        
    def chi2_test(self, dataframe: pd.DataFrame):
        pass
    
    def  select_best(self,dataframe: pd.DataFrame):
        
        selected_feats = []
        # create a Dataframe only for categorical variables
        # categorical_df = pd.get_dummies(dataframe[self.cat_feats])
        categorical_df = dataframe[self.cat_feats]
        
        for feats in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(dataframe[feats].values)
            categorical_df.loc[:,feats] = lbl.transform(dataframe[feats].values)
        
        # select only Top 5 variables 
        selector = SelectKBest(chi2,k=5)
        # give the targetcolumn and the rest of the data to the scalar to fit
        selector.fit(categorical_df,dataframe[self.target_cols])
        # get the indicies of the selected columns
        cols = selector.get_support(indices=True)

        # For display purpose Only
        dfscores = pd.DataFrame(selector.scores_)
        dfcolumns = pd.DataFrame(categorical_df.columns)

        #concat two dataframes for better visualization 
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Features','Score']  #naming the dataframe columns
        featureScores = featureScores.sort_values(by='Score', ascending=False)
        
        utils.bar_plot(
            x_data= featureScores['Features'],
            y_data=featureScores['Score'],
            title="Select_K_Best using CHI2 For Categorical Features",
            x_title="Features",
            y_title="CHI2 Score",
            output_path= os.path.join(self.output_path,"select_k_best_chi2.html")
        )
        
        self.cat_feats = featureScores['Features'].values.tolist()[:self.num_best]
        # drop the columns which did not qualify
        for feats in self.dataframe_d_copy.columns:
            if feats not in self.cat_feats:
                self.dataframe_d_copy = self.dataframe_d_copy.drop(feats,axis=1)
                   
    def _label_encoding(self):
        for feat in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.dataframe[feat].values)
            self.dataframe_d_copy.loc[:,feat] = lbl.transform(self.dataframe[feat].values)
            self.label_encoders[feat] = lbl
        joblib.dump(self.label_encoders, f"{self.output_path}/_label_encoder.pkl")
        return self.dataframe_d_copy
    
    def _binarization(self):
        for feat in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.dataframe[feat].values)
            val = lbl.transform(self.dataframe[feat].values)
            self.dataframe_d_copy = self.dataframe_d_copy.drop(feat,axis=1)
            
            for j in range(val.shape[1]):
                new_col_name = feat + f'__bin_{j}'
                self.dataframe_d_copy[new_col_name] = val[:,j] 
            self.binary_encoders[feat] = lbl
        joblib.dump(self.binary_encoders, f"{self.output_path}/_binary_encoder.pkl")
        return self.dataframe_d_copy
    
    def _one_hot_encoder(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.dataframe[self.cat_feats])
        return ohe.transform(self.dataframe_d_copy[self.cat_feats])
    
    def fit_transform(self):
        if self.enc_types == "label":
            return self._label_encoding()
        elif self.enc_types == "ohe":
            return self._one_hot_encoder()
        elif self.enc_types == "binary":
            return self._binarization()
        else:
            raise Exception("Encoding type not understood")    

if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_csv("input/categorical_features/train.csv")
    cols = [c for c in df.columns if c not in ["id","target"]]
    
    cat_feats = CategoricalFeatures(
        dataframe=df,
        categorical_features=cols,
        enc_type="ohe",
        handle_na=True
    )
    
    output_df = cat_feats.fit_transform()
    print(output_df.head())