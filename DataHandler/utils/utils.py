import numpy as np

def padData(X, X_test, padDim):
    zeros_array = np.zeros((X.shape[0], padDim))
    zeros_array_test = np.zeros((X_test.shape[0], padDim))

    X = np.concatenate((X, zeros_array), axis=1)
    X_test = np.concatenate((X_test, zeros_array_test), axis=1)
    return X, X_test

def testData(pred, Y_test):
    res = (pred-Y_test)**2
    res[np.isnan(res)]=0
    mse = np.mean(res)
    return mse