from numpy import *

from Mapper import Mapper

from MOP import activation

from MLP import sigmoid, dsigmoid
def F_sigmoid(w,x):
    '''
        should return M * L matrix given by the linearization of f(w'x)
    '''
    df = dsigmoid(activation(w,x)) * x   # where activation = (dot(self.w,x))
    return df #* (w - w_0)

from MOP import linear
def F_linear(w,x):
    ''' 
    observation function ?
    ----------------------
    return derivative of w'x
    '''
    return x

class RLS(): # Mapper
    '''
        RLS (Recursive Least Squares) Regressor
        ---------------------------------------

        This is a Bayesian-type model, where
        * w[t] is the (latent) state
        * (x[t],y[t]) is our observation

        It is a recursive version of OLS, if 
        * we suppose y = f(x) = w'x, which has an 'inverse' (derivative) of x 

        Otherwise it can handle non-linear approximations, if 
        * we suppose y = f(x) for non-linearity f, for which there is some inverse f' 
          (inverses exist for sigmoid, etc ...)
    '''

    w = None # the state (e.g., a weight + bias)

    def __init__(self, N_w, N_o, ridge=1., f=linear):

        self.N_i = N_w # @temp: was needed for sth at some point

        # note: N_o=1 assumed for now
        prec = 1./ridge
        self.iR = eye(N_w) * prec
        self.w = random.randn(N_w) * prec

        self.f = f
        if f == linear:
            self.df = F_linear
        elif f == sigmoid:
            self.df = F_sigmoid
        else:
            print "NOT YET IMPLEMENTED"
            exit(1)


    def step(self, (x, y)):
        # Called from Brain/As Mapper
        return self.update(x, y)

    def update(self, x, y):
        '''
            observation (x, y) where y is the desired output of neural network
            return most probable state w
        '''
        yp = self.predict(x)
        P = self.iR
        w = self.w
        ############ NON-LINEARITY ###########################
        x = self.df(w,x)
        ############ RLS FILTER ##############################
        K = dot(P,x)
        w = w + K * (y - yp)
        iB = 1./(1. + dot(dot(x,P),x))
        F = x * iB
        P = P - P * F * K 
        ############ RLS FILTER ##############################
        self.w = w
        self.iR = P
        return self.w

    def predict(self,x):
        '''
            Predict p(y|x) = f(x;w)
            ------------------------
            NOTE: this 'activation' is simply the dot product for OLS, 
            however, for more advanced, e.g., two-layered networks, it's not so straightforward;
            we need to be careful with internal memory variables like z.
        '''
        return self.f(activation(self.w,x)) #return self.f(dot(self.w,x))

    def fire(self,x):
        '''
            Predict
            --------------
        '''
        return self.predict(x)

def demo():
    f=linear #sigmoid
    df=F_linear #F_sigmoid

    N = 100
    D = 2
    L = 1
    X = random.randn(N,2)
    w = array([1.9,-1.2]).T
    y = f(dot(X,w)) + random.randn(N) * 0.1

    print w
    print "#"
    h = RLS(D, L, ridge=10., f=f)
    for i in range(N):
        if i % 10 == 0:
            #print X[i,:], h.w, y[i], h.predict(X[i,:]), (h.predict(X[i,:]) -  y[i])**2
            print h.w , (h.predict(X[i,:]) -  y[i])**2
        h.update(X[i,:],y[i])

    print "============"

if __name__ == '__main__':
    demo()



