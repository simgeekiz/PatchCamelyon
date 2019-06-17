from tensorflow import metrics, local_variables_initializer
from keras.backend import get_session
from sklearn.metrics import roc_auc_score as skroc
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def auc(y_true, y_pred):
    auc = metrics.auc(y_true, y_pred)[1]
    get_session().run(local_variables_initializer())
    return auc

def calculate_auc(y_true, y_pred):
    print("sklearn auc: {}".format(skroc(y_true, y_pred)))
    auc, update_op = tf.compat.v1.metrics.auc(y_true, y_pred)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        print("tf auc: {}".format(sess.run([auc, update_op])))
        
def show(image, now=True, fig_size=(5, 5)):
    image = image.astype(np.float32)
    m, M = image.min(), image.max()
    if fig_size != None:
        plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.imshow((image - m) / (M - m), cmap='gray')
    plt.axis('off')
    if now == True:
        plt.show()