from src.models import efficientnet_model, resnet_model, inception_model
from src.losses import categorical_focal_loss, AdaptiveCategoricalFocalLoss
from src.callbacks import AdaptiveAlphaGammaCallback
from src.data_loader import create_dataset, load_data, split_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from config import *

# Load and split data
images, labels = load_data()
train_images, val_images, train_labels, val_labels = split_data(images, labels)

# Create datasets
train_ds_eff = create_dataset(train_images, train_labels, 'efficientnet')
val_ds_eff = create_dataset(val_images, val_labels, 'efficientnet')
train_ds_res = create_dataset(train_images, train_labels, 'resnet')
val_ds_res = create_dataset(val_images, val_labels, 'resnet')
train_ds_incep = create_dataset(train_images, train_labels, 'inception')
val_ds_incep = create_dataset(val_images, val_labels, 'inception')

# Train models
def train_pipeline(build_fn, train_ds, val_ds, alpha=ALPHA, gamma=GAMMA, initial_lr=LR_INITIAL, fine_tune_lr=LR_FINE_TUNE):
    model = build_fn()
    loss_fn = AdaptiveCategoricalFocalLoss(alpha, gamma)
    adaptive_cb = AdaptiveAlphaGammaCallback(loss_fn, val_ds, CLASS_NAMES)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    
    model.compile(optimizer=Adam(learning_rate=initial_lr),
                  loss=categorical_focal_loss(alpha, gamma),
                  metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_INITIAL, callbacks=[early_stop, checkpoint], verbose=1)
    
    # Fine-tune
    model.trainable=True
    for layer in model.layers[:-20]:
        layer.trainable=False
    model.compile(optimizer=Adam(learning_rate=fine_tune_lr),
                  loss=loss_fn,
                  metrics=['accuracy'])
    history_ft = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE_TUNE, callbacks=[adaptive_cb, early_stop, checkpoint], verbose=1)
    
    return model, history, history_ft

# Train all models
eff_model, eff_hist, eff_ft_hist = train_pipeline(efficientnet_model.build_model, train_ds_eff, val_ds_eff)
res_model, res_hist, res_ft_hist = train_pipeline(resnet_model.build_model, train_ds_res, val_ds_res)
incep_model, incep_hist, incep_ft_hist = train_pipeline(inception_model.build_model, train_ds_incep, val_ds_incep)
