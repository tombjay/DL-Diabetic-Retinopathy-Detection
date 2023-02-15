import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import pandas as pd
from absl import app, flags
from sklearn.utils import shuffle
from input_pipeline.preprocessing import preprocess, augment,squeezer, preprocess_image

@gin.configurable
def load(name, raw_data_dir,data_dir,n_classes,graham):

    if name == "idrid":
            logging.info(f"Preparing dataset {name}...")
            if not graham:
                data_dir = data_dir+"w/o_graham"  
            if not os.path.exists(data_dir):
                logging.info("No Processed Data Found...")
                logging.info("Processing Raw Data...")
                logging.info("Creating Data Directories...")
                
                #Creating the dataset Directories
                os.makedirs(data_dir)
                os.makedirs(os.path.join(data_dir,"train"))
                os.makedirs(os.path.join(data_dir,"test"))
                logging.info(f"Processing training dataset images...")
                
                for image in os.listdir(raw_data_dir + "images/train/"):
                    #Applying Graham Pre-processing
                    img = preprocess_image(raw_data_dir +"images/train/"+image)
                    img.save(os.path.join(data_dir,"train",image))
                logging.info(f"Processing test dataset images...")          
                for image in os.listdir(raw_data_dir + "images/test/"):
                    #Applying Graham Pre-processing
                    img = preprocess_image(raw_data_dir +"images/test/"+image)
                    img.save(os.path.join(data_dir,"test",image))
                logging.info(f"Data Preprocessing Complete...")   
                
            else:

                logging.info("Processed Data Found...") 
   
            #Creating a Balanced data distirbution in the Training and Validation data
            train_df,val_df = train_val_dataframe_creator(raw_data_dir, n_classes)
            test_df = test_dataframe_creator(raw_data_dir, n_classes)
            print (" Number of Training Images:",len(train_df))
            print (" Number of Validation Images:",len(val_df))
            print (" Number of Test Images:",len(test_df))

            #Creating a list of image paths
            train_image = train_df['Image name'].tolist()  
            val_image = val_df['Image name'].tolist() 
            test_image = test_df['Image name'].tolist() 
            
            #Creating a list of labels
            train_labels = train_df['Retinopathy grade'].tolist()  
            val_labels = val_df['Retinopathy grade'].tolist() 
            test_labels = test_df['Retinopathy grade'].tolist() 
            
            #Conversion of image names to data path
            train_images =list(map(lambda x: str(x), [data_dir + "train/" + train+ ".jpg"  for train in train_image]))
            val_images =list(map(lambda x: str(x), [data_dir + "train/" + train + ".jpg" for train in val_image]))
            test_images =list(map(lambda x: str(x), [data_dir + "test/" + test + ".jpg" for test in test_image]))
        
            #Dataset Creation
            ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            ds_val = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
            ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            ds_info = []

            return prepare(ds_train,ds_val, ds_test, ds_info)
        
    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir,
            download=False
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
#Preparing Dataset
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset for custom model
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(600)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.map(squeezer,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    #Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.map(
        squeezer, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    #Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.map(
        squeezer, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train,ds_val, ds_test, ds_info

def train_val_dataframe_creator(raw_data_dir,n_classes):
    
    #Loading the .csv file to a pandas Dataframe
    df = pd.read_csv(raw_data_dir +"labels/train.csv")
    df = df.iloc[:, :2]
    
    #dataframes for each class
    df_train_class0 =df[df["Retinopathy grade"] == 0]
    df_train_class1 =df[df["Retinopathy grade"] == 1]
    df_train_class2 =df[df["Retinopathy grade"] == 2]
    df_train_class3 =df[df["Retinopathy grade"] == 3]
    df_train_class4 =df[df["Retinopathy grade"] == 4]
    
    #Shuffling and repeating the dataframe to make sure all the classes are of the same quantity
    max_size = max(len(df_train_class0),len(df_train_class1),len(df_train_class2),len(df_train_class3),len(df_train_class4))
    min_size = min(len(df_train_class0),len(df_train_class1),len(df_train_class2),len(df_train_class3),len(df_train_class4))
    n_repeats = int(max_size/min_size)+1
    df_train_class0 = shuffle(pd.concat([df_train_class0]*n_repeats)).iloc[:max_size]
    df_train_class1 = shuffle(pd.concat([df_train_class1]*n_repeats)).iloc[:max_size]
    df_train_class2 = shuffle(pd.concat([df_train_class2]*n_repeats)).iloc[:max_size]
    df_train_class3 = shuffle(pd.concat([df_train_class3]*n_repeats)).iloc[:max_size]
    df_train_class4 = shuffle(pd.concat([df_train_class4]*n_repeats)).iloc[:max_size]
    df_trainer = pd.concat([df_train_class0, df_train_class1, df_train_class2, df_train_class3, df_train_class4])
    
    #Balancing the dataframe based on the number of classes
    if n_classes == 2:
        df_result = pd.DataFrame()
        #Class 0 + Class 1 = NRDR; Class 2 + Class 3 + Class 5 = DR
        proportions = [0.25, 0.25, 0.1667, 0.1667, 0.1667]
        for i,proportion in enumerate(proportions):
            n_rows = int(proportion*len(df_trainer)) # number of rows to sample
            df_temp = df_trainer[i*136:(i+1)*136]
            n_sample = min(n_rows, len(df_temp))
            df_temp = df_temp.sample(n=n_rows, replace= True) #sample the rows
            df_result = pd.concat([df_result,df_temp])

        # shuffle the dataframe
        df_result = df_result.sample(frac=1).reset_index(drop=True)
        N_samples = int(0.7*len(df_result))
        df_train = df_result.iloc[:N_samples]
        df_val = df_result.iloc[-(len(df_result)-N_samples):]
        df_train['Retinopathy grade'] =df_train['Retinopathy grade'].map(lambda x: 0 if x < 2 else 1) 
        df_val['Retinopathy grade'] =df_val['Retinopathy grade'].map(lambda x: 0 if x < 2 else 1) 
        
    elif n_classes == 5:
        df_result = pd.DataFrame()
        proportions = [0.20, 0.20, 0.20, 0.20, 0.20]
        for i,proportion in enumerate(proportions):
            n_rows = int(proportion*len(df_trainer)) # number of rows to sampl
            df_temp = df_trainer[i*136:(i+1)*136]
            n_sample = min(n_rows, len(df_temp))
            df_temp = df_temp.sample(n=n_rows, replace= True) #sample the rows
            df_result = pd.concat([df_result,df_temp])

        # shuffle the dataframe
        df_result = df_result.sample(frac=1).reset_index(drop=True)
        N_samples = int(0.7*len(df_result))
        df_train = df_result.iloc[:N_samples]
        df_val = df_result.iloc[-(len(df_result)-N_samples):]
        
    return df_train, df_val      

def test_dataframe_creator(raw_data_dir,n_classes):
    #Loading the .csv file to a pandas Dataframe
    df = pd.read_csv(raw_data_dir +"labels/test.csv")
    df = df.iloc[:, :2]
    
    #No Balancing performed in the test data
    if n_classes ==2:
        df['Retinopathy grade'] =df['Retinopathy grade'].map(lambda x: 0 if x < 2 else 1) 
        test_df = df
    elif n_classes ==5:
        test_df = df
    
    return test_df
