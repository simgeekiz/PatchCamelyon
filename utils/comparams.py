from tensorflow import metrics, local_variables_initializer
from keras.backend import get_session

def auc(y_true, y_pred):
    auc = metrics.auc(y_true, y_pred)[1]
    get_session().run(local_variables_initializer())
    return auc