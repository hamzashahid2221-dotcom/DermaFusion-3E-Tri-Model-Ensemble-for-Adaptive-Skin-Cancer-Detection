import pathlib, tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from config import TRAIN_PATH, CLASS_NAMES, BATCH_SIZE
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as incep_preprocess

def load_data(train_path=TRAIN_PATH, classes=CLASS_NAMES):
    all_images, all_labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = pathlib.Path(train_path)/cls
        images = list(cls_dir.glob('*'))
        all_images.extend([str(img) for img in images])
        all_labels.extend([idx]*len(images))
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    return all_images, all_labels

def split_data(images, labels, test_size=0.2, seed=42):
    return train_test_split(images, labels, test_size=test_size, random_state=seed, stratify=labels)

def preprocess_image(path, label, model_type='efficientnet', classes=CLASS_NAMES):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    if model_type in ['efficientnet','resnet']:
        img = tf.image.resize(img,[224,224])
        img = eff_preprocess(img) if model_type=='efficientnet' else res_preprocess(img)
    elif model_type=='inception':
        img = tf.image.resize(img,[299,299])
        img = incep_preprocess(img)
    return img, tf.one_hot(label, depth=len(classes))

def create_dataset(image_paths, labels, model_type='efficientnet', batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.shuffle(1000)
    ds = ds.map(lambda x,y: preprocess_image(x,y,model_type), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
