'''
    Various activation functions
'''

from numpy import *

def relu(a):
    return (a > 0) * a

def tlu(a):
    return (a > 0.) * 1.

def bound(a,t=0.5):
    return (a >= t) * 1.

def linear(a):
    return a

def sigmoid(a):
    return 1. / (1. + exp(-a))

def dsigmoid(y):
    return sigmoid(y) * (1. - sigmoid(y))

def dtanh(y):
    return 1.0 - y**2.    # dtanh

