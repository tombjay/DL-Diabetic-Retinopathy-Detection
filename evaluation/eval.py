import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
from evaluation.visualization import *
from sklearn.metrics import f1_score,balanced_accuracy_score
import wandb
@gin.configurable
def evaluate(model, checkpoint_dir, ds_test, ds_info, run_paths, model_name, visualize):
    visualize_img = "/Users/thinesh/Desktop/dl-lab-22w-team04/diabetic_retinopathy/Preprocessed_IDRID_dataset/test/IDRiD_072.jpg"
    #Initialize Checkpoints and Checkpoint Manager
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(model=model, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
    
    #Restore Checkpoint into the model 
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    
    #Initialize Metrics
    test_accuracy = tf.keras.metrics.Accuracy(name = "test_accuracy")
    Precision = tf.keras.metrics.Precision(name="test_precision")
    Recall = tf.keras.metrics.Recall(name="test_recall")

    #Model Predicts the output of the test dataset
    logits = model.predict(ds_test)
    
    #Conversion of test labels and Predictions from One hot encoding to integers
    y_test = []
    for _, label in ds_test:
        y_test.extend(tf.argmax(label, -1))
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1)
    y_pred = logits_processing(logits)
    
    #Update the Metrics
    test_accuracy(y_test,y_pred)
    Precision(y_test,y_pred)
    Recall(y_test,y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    if visualize:
        visualizer(model,model_name,visualize_img)
           
    #Logging Metrics into wandb
    log_data = {"Test_Accuracy": test_accuracy.result() * 100,
                    "Precision": Precision.result()*100, "Recall": Recall.result() * 100,
                    "Average_F1_score": f1_macro,
                    "Balanced_Accuracy": balanced_acc *100}
    wandb.log(log_data)
    
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    print("Test set Balanced accuracy: {:.3%}".format(balanced_acc))
    print("Test set Precision: {:.3%}".format(Precision.result()))
    print("Test set Recall: {:.3%}".format(Recall.result()))
    print("Test set F1_score average: {:.3%}".format(f1_macro))
    
    cm = tf.math.confusion_matrix(y_test, y_pred).numpy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    plot_confusion_matrix(cm, classes=[0, 1])
    plt.savefig("/Users/thinesh/Desktop/dl-lab-22w-team04/diabetic_retinopathy/Results/"+model_name+"/confusion_metrics.png")

    return

def visualizer(model,model_name, img_path):
    fig = plt.figure()
    #img_path = "/Users/thinesh/Desktop/dl-lab-22w-team04/diabetic_retinopathy/Preprocessed_IDRID_dataset/test/IDRiD_072.jpg"
    #Process Original image and display
    img = keras.preprocessing.image.load_img(img_path)
    img_a = keras.preprocessing.image.img_to_array(img)/255.
    fig.add_subplot(2, 2, 1)
    plt.imshow(img_a)
    plt.axis('off')
    plt.title("Original Image")
    
    # Preprocess image to be processed by the model
    img_array,_ = preprocess(img_path, 0,2, "custom")
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.shape)
    # GradCam
    grad_cam_mixed_img,colored_heatmap = get_gradcam(model,img_array)
    fig.add_subplot(2, 2, 2)
    plt.imshow(grad_cam_mixed_img)
    plt.axis('off')
    plt.title("GradCAM")
    
    #Guided Backpropogation
    gb_grad_map = get_guided_backprop(model,img_array)
    fig.add_subplot(2, 2, 3)
    plt.imshow(gb_grad_map)
    plt.axis('off')
    plt.title("Guided backprop")
    
    # Guided_grad_cam
    guided_grad_cam_heatmap = guided_grad_cam(gb_grad_map, colored_heatmap)
    fig.add_subplot(2, 2, 4)
    plt.imshow(guided_grad_cam_heatmap)
    plt.axis('off')
    plt.title("Guided GradCAM")
    plt.savefig("/Users/thinesh/Desktop/dl-lab-22w-team04/diabetic_retinopathy/Results/"+model_name+"/vis.jpg")

def ensemble_evaluate(saved_models,ds_test,voting):
    #Initialize Metrics
    test_accuracy = tf.keras.metrics.Accuracy(name = "test_accuracy")
    Precision = tf.keras.metrics.Precision(name="test_precision")
    Recall = tf.keras.metrics.Recall(name="test_recall")
    
    model_1 = tf.keras.models.load_model(saved_models[0])
    model_2 = tf.keras.models.load_model(saved_models[1])
    model_3 = tf.keras.models.load_model(saved_models[2])
    model_4 = tf.keras.models.load_model(saved_models[3])
    model_5 = tf.keras.models.load_model(saved_models[4])
    
    #Model Predicts the output of the test dataset
    logits_1 = model_1.predict(ds_test)
    logits_2 = model_2.predict(ds_test)
    logits_3 = model_3.predict(ds_test)
    logits_4 = model_4.predict(ds_test)
    logits_5 = model_5.predict(ds_test)
    
    #Conversion of test labels and Predictions from One hot encoding to integers
    y_test = []
    for _, label in ds_test:
        y_test.extend(tf.argmax(label, -1))
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1)
    if voting:
        y_model_1= logits_processing(logits_1)
        y_model_2= logits_processing(logits_2)
        y_model_3= logits_processing(logits_3)
        y_model_4= logits_processing(logits_4)
        y_model_5= logits_processing(logits_5)
        y_pred = y_model_1 + y_model_2 + y_model_3 + y_model_4 + y_model_5
        y_pred= np.where(y_pred >= 3, 1, 0)
    else:
        logits = (logits_1+logits_2+logits_3+logits_4+logits_5)/5
        y_pred = logits_processing(logits)
    
    #Update the Metrics
    test_accuracy(y_test,y_pred)
    Precision(y_test,y_pred)
    Recall(y_test,y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(balanced_acc)
    #Logging Metrics into wandb
    log_data = {"Ensemble_Test_Accuracy": test_accuracy.result() * 100,
                    "Ensemble_Precision": Precision.result()*100, "Ensemble_Recall": Recall.result() * 100,
                    "Ensemble_Average_F1_score": f1_macro,
                    "Balanced_Accuracy": balanced_acc *100}
    wandb.log(log_data)
    
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    print("Test set Balanced accuracy: {:.3%}".format(balanced_acc))
    print("Test set Precision: {:.3%}".format(Precision.result()))
    print("Test set Recall: {:.3%}".format(Recall.result()))
    print("Test set F1_score average: {:.3%}".format(f1_macro))
    
    cm = tf.math.confusion_matrix(y_test, y_pred).numpy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    plot_confusion_matrix(cm, classes=[0, 1])
    plt.savefig("/Users/thinesh/Desktop/dl-lab-22w-team04/diabetic_retinopathy/Results/ensemble/confusion_metrics.png")

    return
    
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def logits_processing(logits):
    prediction = tf.argmax(logits, -1)
    y_pred = tf.constant(prediction).numpy()
    y_pred = y_pred.reshape(-1)
    
    return y_pred