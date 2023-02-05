# import files/functions from our code
from . import eda
from . import train
from . import predict
from . import cross_validation
from . import feature_selection

# import python libraries
import os
import yaml
import mlflow
import pandas as pd
from pathlib import Path
from yaml.loader import SafeLoader

def pipeline(cfg : dict):
    # read train and test files
    input_cfg = cfg['input']
    train_df = pd.read_csv(input_cfg['train_file'])
    test_df = pd.read_csv(input_cfg['test_file'])
    
    # create deep copies of dataframes so that we can use original whereever required
    train_df_d_copy = train_df.copy(deep=True)
    test_df_d_copy = test_df.copy(deep=True)
    
    # Step 1 Perform The Automatic EDA
    # eda.eda(input_cfg = input_cfg)
    
    # Step 2 Perform Feature Selection for Categorical and Numerical Features
    feature_selection_cfg = cfg['feature_selection']
    feature_selection_cfg['target_cols'] = input_cfg['target_cols']
    feature_selection_cfg['output_path'] = input_cfg['output_path']
    
    feature_select = feature_selection.FeatureSelection(
        train_df= train_df_d_copy,
        test_df= test_df_d_copy,
        feature_selection_cfg= feature_selection_cfg,
        train=True
    )
    train_df_d_copy,  test_df_d_copy = feature_select.get_df()
    # get the updated dict for feature selection
    cfg['feature_selection'] = feature_select.get_config()
    
    # Step 3 Perform Cross Validation
    cv_cfg = cfg['cross_validation']
    cv_cfg['target_cols'] = input_cfg['target_cols']
    cv = cross_validation.CrossValidation(dataframe = train_df_d_copy,
                                          cv_cfg = cv_cfg)
    train_df_d_copy = cv.split()
    # get the updated dict for feature selection
    cfg['cross_validation'] = cv.get_config()
    
    # Step 4 Model Dispatcher and Training
    # Fill in the train config with the details of above steps 
    train_cfg = cfg['training']
    train_cfg['target_cols'] = input_cfg['target_cols']
    train_cfg['output_path'] = input_cfg['output_path']
    train_cfg['clfs_path'] = []
    train_cfg['cols'] = cfg['feature_selection']['categorical_features']['cols'] \
                        + cfg['feature_selection']['numerical_features']['cols']
    train_cfg['num_folds'] = cv_cfg['num_folds']
    
    train_obj = train.Train(dataframe= train_df_d_copy , train_cfg=train_cfg)
    # get the all the details for mlflow
    # if experiment exists get the experiment id
    # else create a new experiment and get the id
    if cfg['ml_flow']['experiment_exist']:
        experiment = mlflow.get_experiment_by_name(cfg['ml_flow']['experiment_name'])
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(cfg['ml_flow']['experiment_name'])
        
    with mlflow.start_run(run_name=cfg['ml_flow']['run_name'],
                          experiment_id=experiment_id):
        # run the training and get the classifier and classifier path list
        clf , clf_path = train_obj.train()
        # TODO : Think how to log model parameters and metrics.
        metrics = train_obj.get_metrics()
        mlflow.log_metrics(metrics=metrics)
        for (model,model_path) in zip(clf,clf_path):
            model_name = f"{cfg['ml_flow']['experiment_name']}_{Path(model_path).stem}"
            mlflow.sklearn.log_model(model, model_name , registered_model_name= model_name)
            train_cfg['clfs_path'].append(model_path)
        # get the updated dict for feature selection
        cfg['training'] = train_cfg
        # save the entire train config in the output path
        # the data from this file will be given as an input to the predict script
        with open(os.path.join(input_cfg['output_path'],'train_config.yaml'), 'w') as file:
            yaml.dump(cfg, file) 
        mlflow.log_artifacts(input_cfg['output_path'])
    # Step 5 Prediction
    predict.predict(test_df , cfg)
    
if __name__ == "__main__":
    # read/create the config dict for pipeline
    CONFIG = os.environ.get("CONFIG")
    with open(CONFIG) as f:
        cfg = yaml.load(f,Loader=SafeLoader)
    # send the config dict to the pipeline function
    pipeline(cfg=cfg)