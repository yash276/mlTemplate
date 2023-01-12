from . import utils
import pandas as pd
import os
from sklearn.feature_selection import chi2  
from sklearn.feature_selection import SelectKBest

class RegressionDiagnosis:
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        diagnosis_cfg: dict
        ) -> None:
        
        self.train_df = train_df
        self.test_df = test_df
        self.diagnosis_cfg = diagnosis_cfg
        self.goodness_of_fit()
        self.statistical_significance()
        
    def goodness_of_fit(self):
        pass
    
    def statistical_significance(self):
        self.chi2_test()
    
    def chi2_test(self):
        # create a Dataframe only for categorical variables
        categorical_df = self.train_df[self.diagnosis_cfg['cat_cols']]
        # select only Top 5 variables 
        selector = SelectKBest(chi2,k=5)
        # give the targetcolumn and the rest of the data to the scalar to fit
        print(categorical_df.head())
        selector.fit(categorical_df,self.train_df["target"])
        # get the indicies of the selected columns
        cols = selector.get_support(indices=True)

        # For display purpose Only
        dfscores = pd.DataFrame(selector.scores_)
        dfcolumns = pd.DataFrame(categorical_df.columns)

        #concat two dataframes for better visualization 
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Features','Score']  #naming the dataframe columns
        featureScores = featureScores.sort_values(by='Score', ascending=False)
        print(featureScores)
        
        utils.bar_plot(
            x_data= featureScores['Features'],
            y_data=featureScores['Score'],
            title="CHI2 Test For Categorical Features",
            x_title="Features",
            y_title="CHI2 Score",
            output_path= os.path.join(self.diagnosis_cfg['output_path'],"chi2,html")
        )