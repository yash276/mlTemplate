
from sklearn import model_selection
import pandas as pd

if __name__=="__main__":
    # read the train CSV
    df = pd.read_csv("input/train.csv")
    # adding a kfold column with value -1
    df['kfold'] = -1
    # Random Sample the data with df.sample
    # Reset the indices and the drop the Index column 
    df = df.sample(frac=1).reset_index(drop=True)
    # Using Stratified K fold
    # Stratified ensures equal distribution of all classes in each fold
    kf =  model_selection.StratifiedKFold(n_splits=5)

    for fold,(train_idx,val_idx) in enumerate(kf.split(X=df,y=df.target.values)):
        print(len(train_idx),len(val_idx))
        df.loc[val_idx,'kfold'] = fold

    df.to_csv('input/train_folds.csv',index=False)