import tensorflow as tf
import tensorflow.keras as keras
from input_pipeline.preprocessing import *
import numpy as np
import matplotlib.pyplot as plt


#Helper Finction to finc the last covolutional layer
def find_target_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find 4D layer. Cannot apply guided backpropagation")

# Helper Finctions to generate the Gradient Class Activation Maps (GradCAM)
def get_heatmap(model,image_array):
    
    #find the last convolution layer from the model
    last_conv_layer = find_target_layer(model)
    
    #Initialize the model
    grad_model = keras.Model(
        model.inputs, [model.get_layer(last_conv_layer).output,model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, logits = grad_model(image_array)
        index = tf.argmax(tf.squeeze(logits))
        pred = logits[:, index]

    #Find the weights of the neurons in the last layer
    gradients = tape.gradient(pred, last_conv_layer_output)
    mean_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    mean_gradients = tf.reshape(mean_gradients, (-1, 1))

    #Use the weights to produce a weighted map
    pred_weighted = tf.squeeze(
        tf.matmul(last_conv_layer_output, mean_gradients))
    # Eliminate the negative weights and normalize
    heatmap = tf.maximum(pred_weighted, 0) / tf.reduce_max(pred_weighted)

    return heatmap.numpy()

def resize_heatmap(heatmap,img_array):
    #Convert the heatmap from 0 to 1 ---> 0-255
    heatmap = np.uint8(255 * heatmap)
    color_map = plt.cm.get_cmap("jet") #can be choosen as per requirement
    colors = color_map(np.arange(256))[:, :3]
    colored_heatmap = colors[heatmap]

    #Resize the heatmap to size of image
    colored_heatmap = keras.preprocessing.image.array_to_img(colored_heatmap)
    colored_heatmap = colored_heatmap.resize((img_array.shape[1], img_array.shape[2]))
    colored_heatmap = keras.preprocessing.image.img_to_array(colored_heatmap)
    
    return  colored_heatmap / colored_heatmap.max()

def overlay_heatmap(img_array,colored_heatmap,  factor=0.5):
    #overlay the resized heatmap onto the original image
    img_array = tf.squeeze(img_array)
    img_array = img_array / img_array.numpy().max()
    overlayed_img = colored_heatmap * factor + img_array
    overlayed_img = overlayed_img / overlayed_img.numpy().max()
    return overlayed_img.numpy()

def get_gradcam(model,image_array):
    heat_map = get_heatmap(model,image_array)
    colored_heat_map = resize_heatmap(heat_map,image_array)
    generated_heatmap = overlay_heatmap(image_array,colored_heat_map)
    
    return generated_heatmap,colored_heat_map
           
#helper files to perfrom guided backpropogation
@tf.custom_gradient
def guided_relu(x):
    y = tf.nn.relu(x)
    def grad(dy):
        return tf.cast(x > 0, tf.float32) * tf.cast(dy > 0, tf.float32) * dy
    return y, grad

def get_guided_backprop(model,image_array):
    inputs =  tf.convert_to_tensor(image_array)
    last_conv_layer = find_target_layer(model)

    gb_model = keras.Model(
        model.inputs,model.get_layer(last_conv_layer).output)
    layers_list = [layer for layer in model.layers[1:] if hasattr(layer, "activation")]
    for layer in layers_list:
        if layer.activation == keras.activations.relu:
            layer.activation = guided_relu

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        out = gb_model(inputs)

    gradient = tape.gradient(out,inputs)[0]
    # shape of gradient (256,256,3)
    gradient = process_grad(gradient)
    return gradient

def process_grad(grad):
    grad = (grad - tf.reduce_mean(grad)) / (tf.math.reduce_std(grad) + 1e-5)
    grad *= 0.2
    grad += 0.5
    grad = tf.clip_by_value(grad, 0, 1)
    return grad.numpy()

# Guided GradCam, combines the guided back prop and Grad cam
def guided_grad_cam(grad_map, colored_heatmap):
    guided_grad_cam_heatmap = grad_map * colored_heatmap
    return guided_grad_cam_heatmap
