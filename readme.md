# Diabetic Retinopathy Detection
Diabetic retinopathy is a vision-threatening disease that affects millions of people worldwide. Early detection and treatment can significantly improve patient outcomes. This work presents a deep learning-based approach to detect referable and non-referable diabetic retinopathy in digital fundus images.

## Getting started

- Download [IDRID dataset](https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset).
- Install the necessary packages listed in ['requirements.txt'](https://github.tik.uni-stuttgart.de/iss/dl-lab-22w-team04/blob/master/requirements.txt) file by running the command,
  
```
pip install -r requirements.txt
``` 
## How to run the code
### Step 1 - Configuring parameters

The following parameters are to be set in the ['config.gin'](https://github.com/tombjay/DL-Diabetic-Retinopathy-Detection/blob/main/configs/config.gin) file.
    
  - **n_classes** - To choose between 2-class or 5-class classification problem (int).
  - **img_height** & **img_width** - To define dimensions of image presented to the model (int).
  - **model_name** - To choose between different models (int). Available models: "custom", "inception_resnet_v2", "resnet_v2", "inception_v3", "xception" 

### Step 2 - Setting up flags

- To train your first model, run the following command.

```python
python3 main.py
```
- By default, the functionality is to train the 'resnet_v2' model.

- To change functionality, the following flags are to be altered in ['main.py'](https://github.com/tombjay/DL-Diabetic-Retinopathy-Detection/blob/main/main.py) file.
  
  - __train__ - Specify whether to train or evaluate a model (bool).
  - __model_name__ - Specify the model to be trained. #custom, #inception_resnet_v2, #resnet_v2 #inception_v3 #xception (str).
  - __Best_Checkpoint__ - Specify whether to load the best Checkpoint or the latest checkpoint (bool). Only enabled if "train = False".
  - __ensemble__ - Specify whether to evaluate single model or ensemble learning (bool). Only enabled if "train = False".
  - __voting__ - Specify whether to perform voting or overaging for ensemble learning (bool). Only enabled if "train = False" and "ensemble = True".
  - __visualize__ - Specify whether to perform Deep Visualization or Not (bool). Only enabled if "train = False".

### Step 3 - Hyperparameter Optimization (Optional)

Bayesian hyperparameter optimization is performed using the sweeps functionality in wandb. 

  - Open the file ['hyper_parameter_train.py'](https://github.com/tombjay/DL-Diabetic-Retinopathy-Detection/blob/main/hyper_parameter_train.py) and import the required model.
  - Modify the sweep configuration to include the required hyper parameters.
  - Run the file by executing the command,

```python
python3 hyper_parameter_train.py
```
## Evaluation & Results

The models have been evaluated using popular metrics such as test set accuracy, balanced accuracy and F1 score.

Model | Best Accuracy | Average Accuracy | Precision | Recall | F1 score 
--- | --- | --- | --- | --- | --- 
Custom | 81.55% | 80.57% | 80.82 | 92.18 | 0.79 
Inception_Resnet_v2 | 86.41% | 84.79% | 93.1 | 84.37 | 0.86 
Inception_v3 | 83.49% | 82.20% | 85.07 | 89.06 | 0.82
Resnet_v2 | **87.38%** | 85.44% | 93.22 | 85.94 | 0.87
Xception | 86.41% | 84.53% | 90.32 | 87.5 | 0.86
Ensemble(voting) | **88.35%** | - | **91.94** | **89.1** | **0.88**
Ensemble(average) | 87.40% | - | 93.20 | 85.94 | 0.87

### Deep Visualization

![Gradcam_xception](https://github.com/tombjay/DL-Diabetic-Retinopathy-Detection/blob/main/Results/xception/vis.jpg)

For more detailed results kindly have a look in ['Results'](https://github.com/tombjay/DL-Diabetic-Retinopathy-Detection/tree/main/Results) folder.

To view the statistics of our best runs, kindly follow this [link](https://wandb.ai/team_4_dl/Diabetic_Retinopathy_Best%20Runs?workspace=default)
