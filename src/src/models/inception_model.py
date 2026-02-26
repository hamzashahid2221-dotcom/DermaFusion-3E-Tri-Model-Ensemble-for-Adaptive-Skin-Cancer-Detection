from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def build_model(num_classes=3):
    base = InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3))
    base.trainable=False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)
