from __future__ import division

import os
import sys
sys.path.append("..")
sys.path.append("../stainnorm")

import gzip
import h5py
import numpy as np
from tqdm import tqdm
import stain_utils as stu
from keras.utils import HDF5Matrix

def get_chunks(dlen, csize=10000):
    chunks = [(i*csize, (i+1)*csize) for i in range(int(dlen/csize))] 
    if dlen > (int(dlen/csize)*csize):
        chunks += [((int(dlen/csize)*csize), dlen)]
    return chunks

if __name__=="__main__":

    normalizer = 'macenko'
    
    data_dir = '../data/'
    csize = 10000
    
    if normalizer == 'macenko':
        import stainNorm_Macenko as normParam
    elif normalizer == 'reinhard':    
        import stainNorm_Reinhard as normParam
    elif normalizer == 'vahadane':
        import stainNorm_Vahadane as normParam
    else:
        raise ValueError("Unknown normalizer given. Choose from: 'macenko', 'reinhard', 'vahadane'")
        
    print("Selected normalizer:", normalizer.upper())
      
    x_train = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_train_x.h5.gz'), 'rb'), 'x')
    y_train = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_train_y.h5.gz'), 'rb'), 'y')
    
    print("Fitting normalization")

    n = normParam.Normalizer()

    first_x = x_train[10]
    first_y = y_train[10]
    x_train = np.delete(x_train, 10, 0)
    y_train = np.delete(y_train, 10, 0)

    n.fit(first_x)
    
    for frst, lst in get_chunks(len(x_train), csize):
        
        print("Normalizing train chunk: ", frst, lst)

        x_train = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_train_x.h5.gz'), 'rb'), 'x')
        y_train = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_train_y.h5.gz'), 'rb'), 'y')

        x_train = x_train[frst:lst]
        y_train = y_train[frst:lst]

        logf = open('./logs/' + normalizer + '_' + str(frst) + '_' + str(lst) + '_train_skipped.txt', 'w')

        t = () 
        for i, x_ in enumerate(tqdm(x_train)):
            norm_x = None
            try:
                norm_x = n.transform(x_)
            except Exception as e:
                logf.write(str(i) + '; ' + str(e) + '\n')
                y_train = np.delete(y_train, i, 0)
                continue

            t = t + (norm_x,)

        logf.close()
        
        if frst == 0:
            print("DEBUG: Adding the first elements to the arrays")
            bs_inputs = (first_x,) + t
            y_train = np.concatenate((np.array([first_y]), y_train))
        else:
            bs_inputs = t

        x_train = stu.build_stack(bs_inputs)
        
        print("Writing train to h5.gz files")

        h5f = h5py.File(os.path.join(data_dir, normalizer + '/camelyonpatch_level_2_split_train_x_' + normalizer + '_' + str(frst) + '_' + str(lst) + '.h5.gz'), 'w')
        h5f.create_dataset('x', data=x_train, compression='gzip')
        h5f.close()

        h5f = h5py.File(os.path.join(data_dir, normalizer + '/camelyonpatch_level_2_split_train_y_' + normalizer + '_' + str(frst) + '_' + str(lst) + '.h5.gz'), 'w')
        h5f.create_dataset('y', data=y_train, compression='gzip')
        h5f.close()
    
    del x_train
    del y_train
    del bs_inputs
    del t
    
    #################### VALIDATION DATA ####################
    
    x_valid = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5.gz'), 'rb'), 'x')
    y_valid = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5.gz'), 'rb'), 'y')
    
    for frst, lst in get_chunks(len(x_valid), csize):
        
        print("Normalizing valid chunk: ", frst, lst)

        x_valid = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5.gz'), 'rb'), 'x')
        y_valid = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5.gz'), 'rb'), 'y')

        x_valid = x_valid[frst:lst]
        y_valid = y_valid[frst:lst]

        logf = open('./logs/' + normalizer + '_' + str(frst) + '_' + str(lst) + '_valid_skipped.txt', 'w')

        t = () 
        for i, x_ in enumerate(x_valid):
            norm_x = None
            try:
                norm_x = n.transform(x_)
            except Exception as e:
                logf.write(str(i) + '; ' + str(e) + '\n')
                y_valid = np.delete(y_valid, i, 0)
                continue

            t = t + (norm_x,)

        logf.close()

        x_valid = stu.build_stack(t)

        print("Writing valid to h5.gz files")

        h5f = h5py.File(os.path.join(data_dir, normalizer + '/camelyonpatch_level_2_split_valid_x_' + normalizer + '_' + str(frst) + '_' + str(lst) + '.h5.gz'), 'w')
        h5f.create_dataset('x', data=x_valid, compression='gzip')
        h5f.close()

        h5f = h5py.File(os.path.join(data_dir, normalizer + '/camelyonpatch_level_2_split_valid_y_' + normalizer + '_' + str(frst) + '_' + str(lst) + '.h5.gz'), 'w')
        h5f.create_dataset('y', data=y_valid, compression='gzip')
        h5f.close()
    
    del x_valid
    del y_valid
    del t
    
    #################### TEST DATA ####################
    
    print("Loading data with purpose=test")
    
    x_test = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_test_x.h5.gz'), 'rb'), 'x')
    y_test = HDF5Matrix(gzip.open(os.path.join(data_dir, 'camelyonpatch_level_2_split_test_y.h5.gz'), 'rb'), 'y')
    
    print("Normalizing test")

    logf = open('./logs/' + normalizer + '_test_skipped.txt', 'w')
    
    t = () 
    for i, x_ in enumerate(x_test):
        norm_x = None
        try:
            norm_x = n.transform(x_)
        except Exception as e:
            logf.write(str(i) + '; ' + str(e) + '\n')
            norm_x = x_

        t = t + (norm_x,)
        
    logf.close()
    
    x_test = stu.build_stack(t)
    
    print("Writing test to h5.gz files")
    
    h5f = h5py.File(os.path.join(data_dir, normalizer + '/camelyonpatch_level_2_split_test_x_' + normalizer + '.h5.gz'), 'w')
    h5f.create_dataset('x', data=x_test, compression='gzip')
    h5f.close()
    
    h5f = h5py.File(os.path.join(data_dir, normalizer + '/camelyonpatch_level_2_split_test_y_' + normalizer + '.h5.gz'), 'w')
    h5f.create_dataset('y', data=y_test, compression='gzip')
    h5f.close()
    
    
