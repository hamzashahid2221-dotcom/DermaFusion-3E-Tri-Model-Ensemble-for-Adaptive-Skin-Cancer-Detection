import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.data_loader import preprocess_image, create_dataset
from config import CLASS_NAMES

def ensemble_predict(models, val_images, val_labels):
    def preprocess_for_ensemble(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        eff_img = tf.image.resize(img,[224,224])
        res_img = tf.image.resize(img,[224,224])
        incep_img = tf.image.resize(img,[299,299])
        from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
        from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
        from tensorflow.keras.applications.inception_v3 import preprocess_input as incep_preprocess
        eff_img = eff_preprocess(eff_img)
        res_img = res_preprocess(res_img)
        incep_img = incep_preprocess(incep_img)
        return (eff_img, res_img, incep_img), label
    
    ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    ds = ds.map(preprocess_for_ensemble).batch(16).prefetch(tf.data.AUTOTUNE)
    
    eff_prob = models[0].predict(ds.map(lambda x,y:x[0]))
    res_prob = models[1].predict(ds.map(lambda x,y:x[1]))
    incep_prob = models[2].predict(ds.map(lambda x,y:x[2]))
    
    final_prob = (eff_prob + res_prob + incep_prob)/3
    final_pred = np.argmax(final_prob, axis=1)
    
    print(classification_report(val_labels, final_pred, target_names=CLASS_NAMES))
    cm = confusion_matrix(val_labels, final_pred)
    ConfusionMatrixDisplay(cm).plot()
    
# Example usage
# ensemble_predict([eff_model, res_model, incep_model], val_images, val_labels)
