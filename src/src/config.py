import os
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
BASE_PATH = '/kaggle/input/skin-cancer9-classesisic/Skin cancer ISIC The International Skin Imaging Collaboration'
TRAIN_PATH = os.path.join(BASE_PATH, 'Train')
TEST_PATH = os.path.join(BASE_PATH, 'Test')
RESULTS_PATH = '../results/'

# Classes
CLASS_NAMES = ['basal cell carcinoma', 'melanoma', 'squamous cell carcinoma']

# Hyperparameters
BATCH_SIZE = 16
LR_INITIAL = 1e-3
LR_FINE_TUNE = 1e-4
ALPHA = [0.88, 0.75, 1.83]
GAMMA = [2, 2, 4]
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 30
