from numpy import *
from Mapper import Mapper
from kNN import kNN
from MLP import MLPbp
from functions import sigmoid, linear
from RLS import RLS

def jump_size(e,grad_e,t):
    return e * clip(grad_e + 1.,0.,1.)

def project(h,w_origin,jump=0.2,N=100):
    '''
        USE THIS ONE FOR MAD_0.py
    '''
    N_i = h.N_i
    W = random.randn(N,N_i) * jump + w_origin
                        # *NEW* weight penalty
    y = h.predict(W)    # + 0.01*sum(abs(W),axis=1)
    i = argmin(y)
    return W[i,:]

def rwd(e, _e):
    '''
        2nd-order Reward Function
        ----------------------
        Offer second order properties, e.g., the improvement of e, rather than raw e.
        * if we improved, return 0 error (i.e., zero error for any improvement)
        * if not, return the original error
    '''
    if e < _e:
        return 0.
    else:
        return e

def rwd_basic(e, _e):
    return e

class DE(Mapper):
    '''
        Density Estimator (Mapper)
        --------------------------
    '''

    # A classifier to estimate the density
    h = None        

    def __init__(self,N_d,N_o,threshold=5,f=None,f_rwd=rwd_basic):
        #self.h = MLPpf(N_i=len(self.g.get_weight_vector()),N_h=20,N_o=1)
        #self.h = RLS(N_i=len(self.g.get_weight_vector()),N_o=1)
        self.h = kNN(N_i=N_d,N_o=N_o,k=5,n=100)
        #self.h = MLPbp(N_i=N_d,N_h=5,N_o=N_o,f=sigmoid,fo=linear)
        self.f = f # NON-LINEARITY (OPTIONAL)
        #self.h = RLS(N_d,N_o)

        self.learn_rate = 0.1
        self.w = zeros(N_d) # random.randn(N_d) * 0.1        # the state (actually a vector of weights)
        self.t = 0       # temperature
        self.threshold = threshold      # deperation threshold
        self._y = -100. # old reward/error/target
        self.f_rwd = f_rwd
        self.burn_in = 5

    def map(self, w):
        # likelihood function! what observations do we expect given state w
        # this map(self,w) function is used in particular by mad0.py
        yp = self.h.predict(w.reshape(1,-1))
        return yp

    def step(self, (o,y)):
        '''
            The actor carried out y[t] = f(x[t];w[t]) which resulted in error e[t+1]
            1. update critic   x -> e
            2. update actor    x;w;e -> y

            We get observation y,
            and since at the moment the state isn't saved internally here, we can use it here as x
            (x is not the same as the original inputs)
        '''
        #y = y + 0.01 * sqrt(dot(self.x.T,self.x))

        ################################################################################################
        # 1. Update the critic's knowledge map (kNN)
        # (the current self.x (set of weights) produced the result 'y' (e.g., error) in the last round)
        ################################################################################################
        if self.burn_in > 0:
            # not enough points on map yet
            self.h.update(self.w,y) 
            self.burn_in = self.burn_in - 1
            return project(self.h,zeros(self.w.shape),1.,N=1)

        yp = self.h.predict(self.w.reshape(1,-1))
        delta = (y - yp)
        y_ = yp + self.learn_rate * delta
        #print "D", y, delta, y_
        self.h.update(self.w,y_) 

        ################################################################################################
        # 2. Update the actor's knowledge, with queries to the critic
        ################################################################################################

        e_jump = self.f_rwd(y,self._y)
        #print "E", self._y, y, e_jump
        self.t = self.t + 1
        self.w = project(self.h,self.w,jump=e_jump,N=10)
        self._y = y

        return self.w

def summarize(X):
    ''' 
        Summarize
        -----------
        * a very import function, we want to summarize X  into z 
        * i.e., what kind of activity was invoved in X?
    '''
    mu = mean(X,axis=0)
    #dX = X[-1,:] - X[0,:]
    #return dX
    return mu

class DEr(DE):

    '''
        Reinforcement version of DE
    ''' 

    h0 = None

    def __init__(self,N_d,N_o,N_x,threshold=5):
        DE.__init__(self,N_d,N_o,threshold)
        #self.h0 = kNN(N_i=N_x,N_o=N_o,k=5,n=100)
        self.h0 = RLS(N_x,N_o)
        self.emo = -2

    def step(self, (X,e)):

        ################################################################################################
        # 2. Update the OTHER map
        # (if we just tested the null hypothesis)
        ################################################################################################

        z = summarize(X)

        if self.emo == -2:
            print "SUMMARIZE",  z, e
            self.h0.update(z,e) 
            self.t = self.t + 1
            if self.t > 50:
                self.t = 0
                self.emo = 0 
            return self.x
        else:
            # Adjust the error
            e_adj = clip(e - self.h0.predict(z),0.,1.)
            print "ERROR ADJ", e_adj ,"=", e , "-", self.h0.predict(z)
            return DE.step(self,(None,e))


