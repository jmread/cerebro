from numpy import *

from cerebro.functions import relu, linear, sigmoid, dsigmoid, dtanh

class MLP():

    ''' 
        A Multi-Layer Perceptron
        --------------------------------------
    '''

    W_ih = None                # weight matrix
    b_ih = None
    W_ho = None                # weight matrix
    b_ho = None

    def __init__(self, N_i, N_h, N_o, f=sigmoid, fo=linear, density=1.):
        
        self.N_i = N_i
        self.N_h = N_h
        self.N_o = N_o
        # make up some initial weights
        self.W_ih = random.randn(N_i,N_h) * 0.1 * (random.rand(N_i,N_h) <= density)
        self.W_ho = random.randn(N_h,N_o) * 0.1 * (random.rand(N_h,N_o) <= density)
        self.b_ih = zeros(N_h)
        self.b_ho = zeros(N_o)
        self.f = f
        self.fo = fo
        
        # form the initial bias
        #X = random.randn(100,N_i)
        #Y = self.fire(X)
        #self.b = reshape(mean(Y,axis=1).T,(N_o,1))

    def predict(self,x):
        A = dot(x,self.W_ih) + self.b_ih
        Z = self.f(A)          # non-linearity
        y = dot(Z,self.W_ho).T + self.b_ho
        y = self.fo(y)              # (non)-linearity

        # save for back prop!
        self._X = x
        self._Z = Z
        self._Y = y

        return y

    def weight_penalty(self):
        # Frobenius norm
        return sqrt(trace(dot(self.W_ih.T,self.W_ih))) + sqrt(trace(dot(self.W_ho.T,self.W_ho)))

    def to_string(self):
        print "======L1========="
        print self.b_ih
        print self.W_ih
        print "======L2========="
        print self.b_ho
        print self.W_ho
        print "================="

class MLPbp(MLP):
    ''' 
        Multi-Layer Percetron
        ------------------------------------------
        With back propagation for learning weights.
    '''

    _X = None
    _Z = None
    _Y = None

    def update(self, x, y):
        y = nan_to_num(y)
        print "UPDATE", x, y
        self.predict(x.reshape(1,-1))
        self.back_prop(y, learning_rate=0.5)

    def back_prop(self, T, learning_rate=0.01, df=dsigmoid):
        """
        back_propagate
        --------------
        X: a n*d matrix
        Z: a n*h matrix
        Y: a n*L matrix
        T: a n*L matrix
        N: learning rate
        """
        X = self._X
        Z = self._Z
        Y = self._Y
        #print X
        #print Z
        #print Y

        #============= GRADIENT ===========#

        # calculate error terms for output = E_o * sigmoid'(y)
        E_o = (T-Y)
        delta_o = E_o# *  df(Y)               # shape = (1,L)
        
        # calculate error terms for hidden
        #print delta_o.shape
        #print self.W_ho.shape
        E_i = dot(delta_o,self.W_ho.T)              # shape = (1,H)
        delta_i = E_i * df(Z)

        #============= UPDATE =============#

        # update output weights
        grad = dot(Z.T,delta_o)                    # shape = (H,L)
        grad_ = E_o
        self.W_ho = self.W_ho  + learning_rate * grad
        self.b_ho = self.b_ho  + learning_rate * grad_

        # update input weights 
        grad = dot(X.T,delta_i)                     # shape = (D,H)
        grad_ = E_i
        self.W_ih = self.W_ih + learning_rate * grad
        self.b_ih = self.b_ih + learning_rate * grad_

        return sum(E_o)



class MLPpf(MLP):

    ''' 
        Multi-Layer Percetron
        ------------------------------------------------
        With learning and proposals in the weight space.
    '''

    learning_rate = 0.1
    dW = None

    def __init__(self, N_i, N_h, N_o, f=sigmoid, fo=linear, density=1.):
        MLP.__init__(self, N_i, N_h, N_o, f=f, fo=fo, density=density)
        self.wv = zeros(N_i*N_h + N_h*N_o + N_h + N_o)
        self.set_weight_vector(self.wv)
        self.N_i = N_i
        self.N_h = N_h
        self.N_o = N_o

    def get_weight_vector(self):
        return self.wv
        '''
            A Vector Representation of this Network
            ---------------------------------------
            This is necessary so that it may serve as an input vector to another learner.
        '''
        #w1 = self.W_ih.ravel()
        #w2 = self.W_ho.ravel()
        #self.wv[0:len(w1)] = w1
        #l2 = len(w1)+len(w2)
        #self.wv[len(w1):l2] = w2
        #l3 = l2 + len(self.b_ih)
        #self.wv[l2:l3] = self.b_ih
        #self.wv[l3:] = self.b_ho
        #return self.wv

    def set_weight_vector(self, wv):
        # TODO: simply call conv_weight_vector ?
        l1 = self.W_ih.shape[0] * self.W_ih.shape[1]
        self.W_ih = wv[0:l1].reshape(self.W_ih.shape)
        l2 = l1 + self.W_ho.shape[0] * self.W_ho.shape[1]
        self.W_ho = wv[l1:l2].reshape(self.W_ho.shape)
        l3 = l2 + len(self.b_ih)
        self.b_ih = wv[l2:l3].reshape(self.b_ih.shape)
        l4 = l3 + len(self.b_ho)
        self.b_ho = wv[l3:l4].reshape(self.b_ho.shape)

    def conv_weight_vector(self, wv):
        l1 = self.W_ih.shape[0] * self.W_ih.shape[1]
        W_ih = wv[0:l1].reshape(self.W_ih.shape)
        l2 = l1 + self.W_ho.shape[0] * self.W_ho.shape[1]
        W_ho = wv[l1:l2].reshape(self.W_ho.shape)
        l3 = l2 + len(self.b_ih)
        b_ih = wv[l2:l3].reshape(self.b_ih.shape)
        l4 = l3 + len(self.b_ho)
        b_ho = wv[l3:l4].reshape(self.b_ho.shape)
        return W_ih,b_ih,W_ho,b_ho

    def move(self,p=0.1, X=None, reset=False, v_exp=None,p2=0.01):
        '''
            v_exp = vector of exploration
        '''
        v_exp = v_exp + random.randn(len(v_exp)) * 0.1 * p #p2     # add 10% noise
        dW_ih,db_ih,dW_ho,db_ho = self.conv_weight_vector(v_exp)

        # CONSIDER MODDING CONNECTIONS
        n = diag(dot(X.T,X)) + 0.01
        n = (n/sum(n))
        #print "n",n
        j = random.choice(range(self.N_i),p=n)
        #print "j=",j
        k = random.choice(range(self.N_h))
        #dW = random.randn() * p
        #print "dW", dW
        if reset:
            self.W_ih[j,k] = dW_ih[j,k]
            self.b_ih[k] = db_ih[k] * 0.1
        else:
            self.W_ih[j,k] = self.W_ih[j,k] + dW_ih[j,k]
            #print self.b_ih.shape, db_ih.shape
            self.b_ih[k] = self.b_ih[k] + db_ih[k] * 0.1

        # NOTE must be the same as in fire(..)
        A = dot(X,self.W_ih) + self.b_ih
        Z = self.f(A)          # non-linearity
        n = diag(dot(Z.T,Z)) + 0.01
        # TODO PROBABLY NOT NECESSARY NOW
        n = nan_to_num(n)
        #print n
        n = (n/sum(n))
        #print n
        j = random.choice(range(self.N_h),p=n)
        k = random.choice(range(self.N_o))
        #dW = random.randn() * p
        #db = random.randn() * p * 0.1
        #print "dW", dW
        if reset:
            self.W_ho[j,k] = dW_ho[j,k]
            self.b_ho[k] = db_ho[k] * 0.1
        else:
            self.W_ho[j,k] = self.W_ho[j,k] + dW_ho[j,k]
            self.b_ho[k] = self.b_ho[k] + db_ho[k] * 0.1

        #self.weight_decay() 

    def weight_decay(self):
        print "DECAY"
        self.W_ih = self.W_ih * 0.99
        self.W_ho = self.W_ho * 0.99

    def copy(self):
        # TODO: deep copy available?
        N_i,N_h = self.W_ih.shape
        N_h,N_o = self.W_ho.shape
        mlp = MLPpf(N_i,N_h,N_o)
        mlp.W_ih = self.W_ih[:,:]
        mlp.W_ho = self.W_ho[:,:]
        mlp.b_ih = self.b_ih[:]
        mlp.b_ho = self.b_ho[:]
        return mlp

def demo():
    pat = array([
       # X X Y
        [0,0,0,0,0],
        [0,1,1,1,0],
        [1,0,1,1,0],
        [1,1,1,0,1]
    ])

    D = 2
    E = 5
    L = E - D
    X = array(pat[:,0:D],dtype=float)
    Y = array(pat[:,D:E],dtype=float)
    print Y

    print "# SGD #"
    h = MLPbp(D, 4, L)
    for t in range(1000):
        indices = range(4)
        random.shuffle(indices)
        for i in indices:
            x = array([X[i,:]])
            y = array([Y[i,:]])
            #print x,y,
            h.fwd_prop(x)
            error = h.back_prop(y,0.01)
    # test it
    print h.fwd_prop(X)

    print "# GD #"
    h = MLPbp(D, 4, L)
    for t in range(10):
        h.fwd_prop(X)
        error = h.back_prop(Y,0.1)

    print h.fwd_prop(X)

if __name__ == '__main__':
    demo()
