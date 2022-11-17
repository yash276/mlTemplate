
from . import cross_validation
import pandas as pd

if __name__=="__main__":
    # read the train CSV
    df = pd.read_csv("input/multilabel/train.csv")
    cv = cross_validation.CrossValidation(df,
                         target_cols=["attribute_ids"],
                         problem_type="multilabel_classification",
                         multilabel_delimiter=" ",
                         shuffle=True)
    df_split = cv.split()
    print(df_split.head()) 
    print(df_split.kfold.value_counts())
    # df_split.to_csv('input/train_folds.csv',index=False)