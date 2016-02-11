from numpy import *
set_printoptions(precision=3)

def data0(N, w_true=array([1.5, 0.0])):
    '''
        y[t] = 0.5 x[t]
    '''

    X = random.randn(N,2)
    X[:,1] = 1. #NEW
    y = dot(X,w_true.T)
    return X,y

