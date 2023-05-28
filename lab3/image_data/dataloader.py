import os
import pickle
import numpy as np

def load_dl_vectors(dir):
    # read
    with open(os.path.join(dir), 'rb') as f:
        corpus = pickle.load(f)

    # create a numpy 2-D array, with each row as a 150-dim vector
    X = []
    for element in corpus:
        # element['numpy_embedding'] is 2D array, turn it into one D
        element['numpy_embedding'] = element['numpy_embedding'].flatten()
        X.append(element['numpy_embedding'])

    # create the y vector for SVM
    y = []
    for element in corpus:
        y.append(element['label_id'])

    # turn X, y into numpy array
    X = np.array(X)
    y = np.array(y)

    print("--- Load image DL vectors success ---")
    print("X:", X.shape)
    print("y[reference labels]:", y.shape)

    return X, y


def load_image_vectors(dir):
    # read
    with open(os.path.join(dir), 'rb') as f:
        corpus = pickle.load(f)

    X = []
    y = []
    for element in corpus:
        X.append(element['vector'])
        y.append(element['label_id'])

    # turn X, y into numpy array
    X = np.array(X)
    y = np.array(y)

    print("--- Load raw image vectors success ---")
    print("X:", X.shape)
    print("y[reference labels]:", y.shape)

    return X, y
