import os
import gzip
from keras.utils import HDF5Matrix

def load_data(data_dir, purpose='train', limit=None, val_limit=None):
    if purpose == 'train':
        pc_train_x_h5 = gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_train_x.h5.gz'), 'rb')
        pc_train_y_h5 = gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_train_y.h5.gz'), 'rb')
        pc_valid_y_h5 = gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5.gz'), 'rb')
        pc_valid_x_h5 = gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5.gz'), 'rb')
        
        x_train = HDF5Matrix(pc_train_x_h5, 'x')
        y_train = HDF5Matrix(pc_train_y_h5, 'y')
        x_valid = HDF5Matrix(pc_valid_x_h5, 'x')
        y_valid = HDF5Matrix(pc_valid_y_h5, 'y')
        
        if not val_limit and limit:
            val_limit = limit
        
        if limit and limit<len(x_train):
            x_train = x_train[:limit]
            y_train = y_train[:limit]
        
        if val_limit and val_limit<len(x_valid):
            x_valid = x_valid[:val_limit]
            y_valid = y_valid[:val_limit]

        return x_train, y_train, x_valid, y_valid

    elif purpose == 'test':
        pc_test_x_h5 = gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_test_x.h5.gz'), 'rb')
        pc_test_y_h5 = gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_test_y.h5.gz'), 'rb')

        x_test = HDF5Matrix(pc_test_x_h5, 'x')
        y_test = HDF5Matrix(pc_test_y_h5, 'y')

        return x_test, y_test
    
    else:
         print('Please define the purpose. options: "train, test"' )