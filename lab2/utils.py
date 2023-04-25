import pandas as pd

def load_cora(return_validation = False):
    train = pd.read_pickle('./processed/corax_train.pkl')
    test = pd.read_pickle('./processed/corax_test.pkl')
    validation = pd.read_pickle('./processed/corax_validation.pkl')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_validation = validation.iloc[:, :-1]
    y_validation = validation.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    
    if return_validation == True:
        return X_train, y_train, X_validation, y_validation, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test