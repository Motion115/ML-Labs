# import sklearn's linear models
from sklearn.linear_model import SGDRegressor, LassoCV, RidgeCV, LinearRegression
# import sklearn's linear model with feature selection
from sklearn.linear_model import LarsCV
# import sklearn's Bayesian regression
from sklearn.linear_model import ARDRegression
# import sklearn's support vector regression
from sklearn.svm import SVR
# import sklearn's nearest neighbor regression
from sklearn.neighbors import KNeighborsRegressor
# import sklearn's decision tree regression
from sklearn.tree import DecisionTreeRegressor
# import sklearn's ensemble regression
from sklearn.ensemble import GradientBoostingRegressor
# import sklearn's neural network regression
from sklearn.neural_network import MLPRegressor

# import sklearn's metrics
from sklearn.metrics import mean_squared_error, r2_score
# define a function for adjusted r2_score
def adj_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k)

import pandas as pd
from sklearn.preprocessing import StandardScaler
def load_data(train_file, test_file, is_normalize=True):
    FILE_DIR = "./lab1/dataset/"
    # load the data
    train_file = pd.read_csv(FILE_DIR + train_file)
    test_file = pd.read_csv(FILE_DIR + test_file)
    # extract the house_price column for y
    train_y = train_file["house_value"]
    test_y = test_file["house_value"]
    # drop the house_price column for X
    train_file.drop("house_value", axis=1, inplace=True)
    test_file.drop("house_value", axis=1, inplace=True)
    # in any case, return it as numpy array
    train_X = train_file.to_numpy()
    test_X = test_file.to_numpy()
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    if is_normalize:
        # normalize the data with sklearn scaler
        scaler = StandardScaler()
        train_file = scaler.fit_transform(train_X)
        test_file = scaler.transform(test_X)

    return train_X, train_y, test_X, test_y


# main
if __name__ == "__main__":
    X, y, test_X, test_y = load_data("train_set.csv", "test_set.csv", True)
    # print the size of X, y, test_X, test_y
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("test_X shape: ", test_X.shape)
    print("test_y shape: ", test_y.shape)

    # use SGDRegressor
    sgd = LinearRegression()
    sgd.fit(X, y)
    sgd_y_pred = sgd.predict(test_X)
    sgd_r2 = r2_score(test_y, sgd_y_pred)
    sgd_adj_r2 = adj_r2_score(sgd_r2, test_y.shape[0], test_X.shape[1])
    print("SGDRegressor r2 score: ", sgd_r2)
    print("SGDRegressor adjusted r2 score: ", sgd_adj_r2)

    # use gradient boosting regressor
    gbr = GradientBoostingRegressor()
    gbr.fit(X, y)
    gbr_y_pred = gbr.predict(test_X)
    gbr_r2 = r2_score(test_y, gbr_y_pred)
    gbr_adj_r2 = adj_r2_score(gbr_r2, test_y.shape[0], test_X.shape[1])
    print("GradientBoostingRegressor r2 score: ", gbr_r2)
    print("GradientBoostingRegressor adjusted r2 score: ", gbr_adj_r2)




