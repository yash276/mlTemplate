from sklearn.metrics import  r2_score

# TODO return matrices depending on the model type like linear or logistic
def metrics(y_true,preds)-> dict:
        score = {}
        score['r2_score'] = r2_score(y_true,preds)
        
        return score