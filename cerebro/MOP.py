from numpy import *

from functions import relu, linear

def activation(wv,x):
    ''' activation function where wv also encodes bias '''
    #return dot(wv[0:-1].T,x) + wv[-1]
    return dot(wv.T,x)

def static_fire(wv,x,f):
    return f(activation(wv,x))         # non-linearity

class MOP():

    ''' 
        Multi-Output Percetron
        --------------------------------------
    '''

    W = None                # weight matrix
    #b = None

    def __init__(self, N_i, N_o, f=relu):
        
        # make up some initial weights
        self.W = random.randn(N_i,N_o) * 0.5
        self.b = zeros(N_o)
        self.f = f
        self.N_i = N_i
        self.N_o = N_o

        # form the initial bias
        #X = random.randn(100,N_i)
        #Y = self.fire(X)
        #print mean(Y,axis=1)
        #self.b = reshape(mean(Y,axis=1).T,(N_o,))
        #print self.b

    def predict(self,x):

        #print self.W
        #print x
        #print "X, W'", x, self.W.T
        A = dot(self.W.T,x) + self.b
        #print "A", A
        #print dot(self.W.T,x)
        #print dot(x,self.W)
        #exit(1)
        y = self.f(A)         # non-linearity
        #print self.W, x, "=>", y

        return y

    def weight_penalty(self):
        # Frobenius norm
        return sqrt(trace(dot(self.W.T,self.W)))

    def to_string(self):
        print "======L1========="
        print self.b
        print self.W


class MOPbp(MOP):

    dW = None                # weight matrix

    def __init__(self, N_i, N_o, f=relu, wv_init=None):
        MOP.__init__(self,N_i,N_o,f)
        self.dW = random.randn(N_i,N_o) * 0.1

    def back_prop(self, e, lamda=0.01):
        '''
            Training
        '''
        dW = lamda * -e
        self.W = self.W + dW


class MOPpf(MOP):

    ''' 
        Multi-Output Percetron
        --------------------------------------
        Extended such that proposals can be made in the weight space.
    '''

    wv = None
    use_bias = False        # <--- todo, make simpleMOP instead

    def check_(self):
        return (sum(self.W != self.wv.reshape(self.W.shape)) == 0)  

    def __init__(self, N_i, N_o, f=relu, use_bias=False, wv_init=None):
        MOP.__init__(self,N_i,N_o,f)
#        if wv_init is not None:
#            self.wv = wv_init
#        else:
#            self.wv = zeros(N_i*N_o)           # weight vector
        #self.W = self.wv.reshape(N_i,N_o)  # a view of wv
        self.use_bias = use_bias
        self.wv = self.init_weight_vector()
        self.set_weight_vector(self.wv)

    def init_weight_vector(self):
        '''
            Create a new weight vector
            --------------------------
        '''
        if self.use_bias:
            w = zeros(self.N_i*self.N_o + self.N_o)
            w[:-self.N_o] = self.W.reshape(self.N_i*self.N_o)
            w[-self.N_o:] = self.b.reshape(self.N_o)
            return w
        else:
            self.b = zeros(self.N_o)
            w = zeros(self.N_i*self.N_o)
            w[:] = self.W.reshape(self.N_i*self.N_o)
            return w

    def get_weight_vector(self):
        '''
            A Vector Representation of this Network
            ---------------------------------------
            This is necessary so that it may serve as an input vector to another learner.
        '''
        return self.wv

    #    N_i,N_o = self.W.shape
    #    # note: bias not copied!
    #    #w = zeros(N_i*N_o+N_o)          # todo -- need to copy in memory?
    #    self.wv = self.W.ravel(N_i*N_o)
    #    #w[:-N_o] = self.W.reshape(N_i*N_o)
    #    #w[-N_o:] = self.b
    #    return self.wv

    def set_weight_vector(self, wv):
        self.wv[:] = wv[:]
        self.W = self.wv.reshape(self.W.shape) # TODO SHOULD NOT BE NEEDED!

    def propose(self,X):
        N_i,N_o = self.W.shape
        n = diag(dot(X.T,X))+0.01
        n = (n/sum(n))
        j = random.choice(range(N_i),p=n)
        k = random.choice(range(N_o))
        return j,k

    def back_prop(self, e, lamda=0.01):
        '''
            Training
            NOTE: THIS FUNCTION ALREADY EXISTS IN MOPbp
        '''
        dW = lamda * -e
        self.set_weight_vector(self.wv + dW)
        if not self.check_():
            print "DID NOT PASS TEST"
            print self.W
            print self.wv
            exit(1)


    def move(self,p=0.2, X=None, reset=False, v_exp=None, p2=0.1):
        '''
            v_exp = vector of exploration
        '''
        #j,k = self.propose(X)

        #dW = v_exp.reshape(self.W.shape)

        if reset:
            #self.W[j,k] = dW[j,k]
            self.wv[:] = v_exp[:]
        else:
            #self.W[j,k] = self.W[j,k] + dW[j,k]
            self.wv[:] = self.wv[:] + v_exp[:]

    def copy(self):
        mop = MOPpf(self.W.shape[0],self.W.shape[1])
        mop.W = self.W[:,:]
        #mop.b = self.b[:]
        mop.dW = self.dW[:,:]
        return mop
