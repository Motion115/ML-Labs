import pandas as pd
import os
# import logistics regression in sklearn
from sklearn.linear_model import LogisticRegression

# switch to current file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# the cora features
data = pd.read_pickle('./corpus/corax_features.pkl')
label = pd.read_pickle('./corpus/corax_labels.pkl')
# label from one-hot to single label
label = label.argmax(axis=1)

# split data and label in a uniform way
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)

# train the model
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
# predict the test data
y_pred = clf.predict(X_test)
# calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
