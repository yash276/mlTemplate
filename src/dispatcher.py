from sklearn import ensemble
from sklearn import linear_model

models = {
    "randomforest" : ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,verbose=2),
    "extratrees" : ensemble.ExtraTreesClassifier(n_estimators=200,n_jobs=-1,verbose=2),
    "logistic" : linear_model.LogisticRegression(),
    "linear" : linear_model.LinearRegression(),
}