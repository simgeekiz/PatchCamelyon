import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
# from keras.applications.mobilenet import preprocess_input
from keras.utils import to_categorical
from PIL import Image
import random
from random import sample, getrandbits
from .Augmentation import Augmentation


#Data Generator to efficiently load and preprocess data for training the classifier

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data, labels, batch_size=32, aug=True, dim=(96, 96),
        n_channels=3, n_classes=2, shuffle=True):
        '''Initialization'''
        self.dim = dim
        self.batch_size = batch_size
        self.aug = aug
        self.data = data
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''Generates data containing batch_size samples''' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes))

        # printing the progress
#         print(str(int((int(list_IDs_temp[0])/len(self.list_IDs))*100)) + '%')

        for i, ID in enumerate(list_IDs_temp):
            # preprocessing
            img_arr = self.data[ID]
            img = array_to_img(img_arr)
            if self.dim[0] != 96 and self.dim[1] != 96:
                img = img.resize((self.dim), Image.ANTIALIAS)
            img.load()

            if self.aug and random.randint(0,1) < 1:
                X[i] = Augmentation(np.asarray(img, dtype=np.uint8))/255
            else:
                X[i] = np.asarray(img, dtype=np.uint8)/255

            # Store target label(one-hot-encoding)
            y[i] = to_categorical(self.labels[str(ID)], num_classes=self.n_classes)

        return X, y
    
def multiple_batch_generator(generator_num, **kwargs):
    """A generator to work with multiple inputs models
    
    We create a model with a list of multiple input layers when 
    we use :func:`keras.layers.concatenate`. However,
    :class:`batch_generator.BatchGenerator` returns a single tuple 
    with two arrays, which does not fit to a model with a multiple 
    input layers. Thus, with this generator, we create the necessary 
    input for such models.
    
    Arguments:
        generator_num {int} -- number of generators should be created
        \**kwargs -- See :class:`batch_generator.BatchGenerator`
        
    Yields:
        ([ndarray,...,ndarray], ndarray) -- in the tuple; list contains feature arrays from each generator, array out of the list contains the label set
    """
    #generators_list = [BatchGenerator(**kwargs, shuffle=False) for i in range(generator_num)]
    gen = DataGenerator(**kwargs, shuffle=False)
    
    i = 0
#     while i < gen.__len__():
    while True:
        nx = gen.__getitem__(i)
        Xy_list = [nx]*generator_num
        i += 1
        yield [Xy[0] for Xy in Xy_list], Xy_list[0][1]
