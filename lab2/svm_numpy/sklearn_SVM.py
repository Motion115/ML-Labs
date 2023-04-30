# Description: SVM using sklearn

# to solve import issue
import sys, os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR))
from utils import load_cora

import pandas as pd
# import SVM from sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import load_cora

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cora()
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

