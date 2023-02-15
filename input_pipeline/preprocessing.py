import gin
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np

@gin.configurable
def preprocess(image, label,n_classes, model_name):
    """Dataset preprocessing: Image processing,normalizing,resizing"""
    # convert image path to opencv image- suitable for image preprocessing
    image = tf.image.decode_image(tf.io.read_file(image),channels=3,expand_animations = False)
    image = tf.cast(image, tf.float32)
    #converts image from 0-255 to -1 to +1
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    
    #converting label to one hot coding
    label = tf.one_hot(label,depth= n_classes)
    return image,label

@gin.configurable
def preprocess_image(image_path, img_height, img_width,graham):
    """Dataset preprocessing: Image processing,normalizing,resizing"""
    
    #crop the unwanted areas in the sides of the image and resize as per requirement
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (img_height, img_width))
    if graham:
        # adding the original image with the smoothed image (gaussian_blur) to highlight the blood vessels, exudates, aneurysms and  hemorrhages. 
        # kernel sizes set to zero as the kernel values are dependent on the sigmaX(standard_deviation) value. sigmaX = 10 ( used by Graham) 
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX =10), -4, 128)
     
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb, 'RGB')
       
    return img

def crop_image_from_gray(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

#Applying Data Augmentation to the Dataset
def augment(image, label,img_height = 256, img_width = 256):
    """Data augmentation"""
    # refered to https://www.tensorflow.org/api_docs/python/tf/image
    image = tf.image.random_flip_left_right(image) 
    image = tf.image.random_flip_up_down(image)
    
    #image = tf.image.resize(tf.image.random_crop(image, size =(int(img_height*0.85),int(img_width*0.85),3)), size =(img_height,img_width))

    return image, label

def squeezer(image,label):
    image = tf.squeeze(image)

    return image,label

