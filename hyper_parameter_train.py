import gin
import logging
import tensorflow as tf
from absl import app, flags
import os
import wandb
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc 
from models.architectures import customModel, inception_resnet_v2, inception_v3
#ds_train, ds_val, ds_test, ds_info = datasets.load()


def train_function():
    with wandb.init() as run:
        gin.clear_config()

        # Hyperparameters for tuning
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # Generate folder structures
        run_paths = utils_params.gen_run_folder()

        # Set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # Gin-config
        gin.parse_config_files_and_bindings(['/home/RUS_CIP/st176497/dl-lab-22w-team04/diabetic_retinopathy/configs/config.gin'], bindings) # change path to absolute path of config file
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup wandb

        # setup pipeline
        print("========================Loading Dataset=================================")


        ds_train, ds_val, ds_test, ds_info = datasets.load()
        
        model = customModel()
            
        print("========================Model Loaded===================================")

        
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths,"custom")
        for _ in trainer.train():
            continue
        checkpoint_dir = "diabetic_retinopathy/checkpoints/best_ckpt/" + "custom"
        evaluate(model,
                    checkpoint_dir,
                    ds_test,
                    ds_info,   
                    run_paths)
        


# Hyperparameter configuration for WandB sweep
sweep_config = {
    "method": "bayes",
    'metric': {
        'name': 'Test_Accuracy',
        'goal': 'maximize',
    }, 
    "parameters": {
        "Trainer.learning_rate": {
            "values": [0.0001, 0.001, 0.01]
        },
        "customModel.dense_units": {
            "values": [16, 32, 64,128]
        },
        "customModel.dropout_rate": {
            'distribution': 'uniform',
        'min': 0.2,
        'max': 0.6,
        }
    }
}

# Set sweep ID
sweep_id = wandb.sweep(sweep_config, project='Diabetic Retinopathy')

# Start WandB agent
wandb.agent(sweep_id, function=train_function)