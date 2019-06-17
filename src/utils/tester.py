import os
import gzip

import numpy as np
from IPython import display

import time

from keras import backend as K
from keras.callbacks import Callback
from keras.models import Model, load_model, model_from_json
from keras.optimizers import SGD, Adam
from keras.utils import HDF5Matrix
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate 
from keras.applications.vgg19 import VGG19

from keras.models import load_model

import tensorflow as tf

import sys
sys.path.append("..")

from utils.load_data import load_data
from utils.preprocess import DataGenerator
from utils.comparams import calculate_auc, auc
from utils.callbacks import PlotCurves

def run_test(model_path, data_dir='data/macenko/', epoch=15, batch_size=128, input_shape=(96, 96)):
    
    model = load_model(model_path, custom_objects={'auc': auc})
    
    x_test, y_test_true = load_data(data_dir, purpose='test', norm='macenko')
    
    # indexes
    test_id = np.arange(len(x_test))

    partition = {}
    partition['test'] = test_id

    test_labels = {str(i) : y_test_true[i].flatten()[0] for i in test_id}

    # Parameters for generators
    params = {
        'dim': input_shape,
        'batch_size': batch_size,
        'n_classes': 2,
        'aug': False,
        'shuffle': False
    }

    # Generators
    test_generator = DataGenerator(partition['test'], x_test, test_labels, **params)

    preds = model.predict_generator(test_generator)

    true_labels = np.array(y_test_true).flatten()
    pred_labels = np.array([p[1] for p in preds])
    calculate_auc(true_labels, pred_labels)