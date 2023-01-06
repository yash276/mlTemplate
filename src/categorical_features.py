from sklearn import preprocessing

class CategoricalFeatures:
    def __init__(
        self,
        dataframe,
        cat_feats_cfg: dict
        ):
        """_summary_

        Args:
            dataframe (_type_): _description_
            categorical_features (_type_): _description_
            enc_type (_type_): _description_
            handle_na (bool, optional): _description_. Defaults to False.
        """
        # the dataframe should not have NA values
        self.dataframe = dataframe
        
        self.cat_feats = cat_feats_cfg['categorical_features']
        self.enc_types = cat_feats_cfg['enc_type']
        self.handle_na = cat_feats_cfg['handle_na']
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None
        
        if self.handle_na:
            for feat in self.cat_feats:
                self.dataframe.loc[:,feat] = self.dataframe.loc[:,feat].astype(str).fillna("-9999999")
        self.dataframe_d_copy = self.dataframe.copy(deep=True)
                
    def _label_encoding(self):
        for feat in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.dataframe[feat].values)
            self.dataframe_d_copy.loc[:,feat] = lbl.transform(self.dataframe[feat].values)
            self.label_encoders[feat] = lbl
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
        
        return self.dataframe_d_copy
    
    def _one_hot_encoder(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.dataframe[self.cat_feats].values)
        
        return ohe.transform(self.dataframe_d_copy[self.cat_feats].values)
    
    def fit_transform(self):
        if self.enc_types == "label":
            return self._label_encoding()
        elif self.enc_types == "ohe":
            return self._one_hot_encoder()
        elif self.enc_types == "binary":
            return self._binarization()
        else:
            raise Exception("Encoding type not understood")
    
    def transform(self,dataframe):
        if self.handle_na:
            for feat in self.cat_feats:
                dataframe.loc[:,feat] = dataframe.loc[:,feat].astype(str).fillna("-9999999")
        if self.enc_types == "label":
            for col, lbl in self.label_encoders.items():
                dataframe.loc[:,col] = lbl.transform(dataframe[col].values)
                
        elif self.enc_types == "binary":
            for col, lbl in self.binary_encoders.items():
                val = lbl.transform(self.dataframe[feat].values)
                dataframe = dataframe.drop(feat,axis=1)
            
                for j in range(val.shape[1]):
                    new_col_name = feat + f'__bin_{j}'
                    dataframe[new_col_name] = val[:,j]
        
        return dataframe      

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