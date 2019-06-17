
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge

def build_vgg16(input_shape=(96,96,3)):
    
    vgg16 = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    x = vgg16.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    y = Dense(2, activation='softmax')(x)
    model = Model(inputs=vgg16.input, outputs=y)
    
    for layer in vgg16.layers:
        layer.trainable = False

    return model

def build_nasnet(input_shape=(96,96,3)):
    
    inputs = Input(input_shape)
    nasnet = NASNetLarge(input_tensor=inputs, weights='imagenet', include_top=False)
    x = nasnet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    y = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=y)
    
    for layer in nasnet.layers:
        layer.trainable = False

    return model

def build_vgg19(input_shape=(96,96,3)):
    
    vgg19 = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)
    x = vgg19.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    y = Dense(2, activation='softmax')(x)
    model = Model(inputs=vgg19.input, outputs=y)
    
    for layer in vgg19.layers:
        layer.trainable = False

    return model

def build_inception_resnet(input_shape=(96,96,3)):
    
    inputs = Input(input_shape)
    inception = InceptionResNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    x = inception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    y = Dense(2, activation='softmax')(x) # sigmoid instead of softmax to have independent probabilities
    model = Model(inputs=inception.input, outputs=y)
    
    for layer in inception.layers:
        layer.trainable = False

    return model

