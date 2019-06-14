import os
import gzip
from keras.utils import HDF5Matrix

def load_data(data_dir, purpose='train', limit=None, val_limit=None, norm=None, is_gzip=False):

    norm = '_' + norm if norm else ''
    gz = '.gz' if is_gzip or norm else ''

    if purpose == 'train':

        pc_train_x_h5 = os.path.join(data_dir, 'camelyonpatch_level_2_split_train_x' + norm + '.h5' + gz)
        pc_train_y_h5 = os.path.join(data_dir, 'camelyonpatch_level_2_split_train_y' + norm + '.h5' + gz)
        pc_valid_y_h5 = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y' + norm + '.h5' + gz)
        pc_valid_x_h5 = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x' + norm + '.h5' + gz)

        if is_gzip and not norm:
            pc_train_x_h5 = gzip.open(pc_train_x_h5, 'rb')
            pc_train_y_h5 = gzip.open(pc_train_y_h5, 'rb')
            pc_valid_y_h5 = gzip.open(pc_valid_y_h5, 'rb')
            pc_valid_x_h5 = gzip.open(pc_valid_x_h5, 'rb')

        x_train = HDF5Matrix(pc_train_x_h5, 'x')
        y_train = HDF5Matrix(pc_train_y_h5, 'y')
        x_valid = HDF5Matrix(pc_valid_x_h5, 'x')
        y_valid = HDF5Matrix(pc_valid_y_h5, 'y')

        if norm and (not limit or limit > x_train.shape[0]):
            limit = x_train.shape[0]

        if not val_limit and limit:
            val_limit = limit

        if not val_limit or val_limit > len(x_valid):
            val_limit = x_valid.shape[0]

        if limit:
            x_train = x_train[:limit]
            y_train = y_train[:limit]

        if val_limit:
            x_valid = x_valid[:val_limit]
            y_valid = y_valid[:val_limit]

        return x_train, y_train, x_valid, y_valid

    elif purpose == 'test':

        pc_test_x_h5 = os.path.join(data_dir, 'camelyonpatch_level_2_split_test_x' + norm + '.h5' + gz)
        pc_test_y_h5 = os.path.join(data_dir, 'camelyonpatch_level_2_split_test_y' + norm + '.h5' + gz)

        if is_gzip and not norm:
            pc_test_x_h5 = gzip.open(pc_test_x_h5, 'rb')
            pc_test_y_h5 = gzip.open(pc_test_y_h5, 'rb')

        x_test = HDF5Matrix(pc_test_x_h5, 'x')
        y_test = HDF5Matrix(pc_test_y_h5, 'y')

        if norm and (not limit or limit > x_test.shape[0]):
            limit = x_test.shape[0]

        if limit:
            x_test = x_test[:limit]
            y_test = y_test[:limit]

        return x_test, y_test

    else:
         print('Please define the purpose. options: "train", "test"')
