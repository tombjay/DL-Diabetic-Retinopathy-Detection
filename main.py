import gin
import logging
import tensorflow as tf
from absl import app, flags
import os
import wandb
from train import Trainer
from evaluation.eval import evaluate, ensemble_evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc 
from models.architectures import customModel, inception_resnet_v2, xception, inception_v3, vgg_like, resnet_v2

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train',False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('model_name', "inception_v3", 'Specify whether to train custom, inception_resnet_v2, xception, inception_v3 or resnet_v2.')
flags.DEFINE_boolean('Best_Checkpoint', True, 'Specify whether to load the best Checkpoint or the latest checkpoint.')
flags.DEFINE_boolean('ensemble',False, 'Specify whether to evaluate single model or ensemble learning.')
flags.DEFINE_boolean('voting',True, 'Specify whether to perform voting or overaging for ensemble learning.')
flags.DEFINE_boolean('visualize',False, 'Specify whether to perform Deep Visualization or Not')
@gin.configurable
def main(argv): 
    
    # generate folder structures
    run_paths = utils_params.gen_run_folder()   
    
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'],logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['/Users/thinesh/Desktop/dl-lab-22w-team04/diabetic_retinopathy/configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    #wandb Initialization
    wandb.init(project="Diabetic Retinopathy", entity="team_4_dl",sync_tensorboard=True)
    
    # setup pipeline
    logging.info('Loading Dataset..............')
    ds_train, ds_val, ds_test, ds_info = datasets.load()
    
    # model
    if FLAGS.model_name == "custom":
        #model = ResNet34(input_shape=[256, 256, 3])
        model = customModel()
        model.summary()
        #model =vgg_like()
        
    elif FLAGS.model_name == "inception_resnet_v2":
        model = inception_resnet_v2()
        
    elif FLAGS.model_name == "resnet_v2":
        model = resnet_v2()
        
    elif FLAGS.model_name == "inception_v3":
            model = inception_v3()
        
    elif FLAGS.model_name == "xception":
        model = xception()
       
    logging.info('Loading Model..............')

    
    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths,FLAGS.model_name)
        for _ in trainer.train():
            continue
     
    else:
        if FLAGS.ensemble:
            saved_models= []
            saved_models.append("diabetic_retinopathy/checkpoints/best_model/custom")
            saved_models.append("diabetic_retinopathy/checkpoints/best_model/inception_resnet_v2")
            saved_models.append("diabetic_retinopathy/checkpoints/best_model/resnet_v2")
            saved_models.append("diabetic_retinopathy/checkpoints/best_model/inception_v3")
            saved_models.append("diabetic_retinopathy/checkpoints/best_model/xception")
            ensemble_evaluate(saved_models,ds_test,FLAGS.voting)
        else:
            if FLAGS.Best_Checkpoint:
                #Loads the Best Checkpoint into the model
                checkpoint_dir = "diabetic_retinopathy/checkpoints/best_ckpt/" + FLAGS.model_name
            else:
                #Loads the Last Checkpoint into the model
                checkpoint_dir = run_paths["path_ckpts_train"]
            
            evaluate(model,
                    checkpoint_dir,
                    ds_test,
                    ds_info,   
                    run_paths,FLAGS.model_name,FLAGS.visualize)
            

if __name__ == "__main__":
    app.run(main)