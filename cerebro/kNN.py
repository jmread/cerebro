from numpy import *
from sklearn.neighbors import KNeighborsRegressor

class kNN():
    '''
        kNN classifier
        -------------
    '''

    def __init__(self,N_i,N_o,k=5,n=20):
        # note: N_o=1 assumed for now
        self.N_i = N_i
        self.n = n
        self.i = 0
        self.k = k
        self.X = zeros((self.n,N_i))
        self.y = zeros((self.n))
        self.h = KNeighborsRegressor(n_neighbors=k,weights='distance')#='distance')
        self.c = 0
        #self.error_rate = 0

    def predict(self,x):
        '''
            Predict
            --------------
        '''

        if self.c < 1.:
            print "[Warning!] No training examples!"
            return 0.0
        elif self.c <= self.k:
            dist,ind = self.h.kneighbors(self.X[0:self.c],n_neighbors=1)
            i_max = argmax(ind)
            return self.y[i_max]

        return self.h.predict(x)#.reshape(1,-1))

#    def samples_X(self):
#        ''' return samples of the WEIGHTS '''
#        if self.c <= 0:
#            return self.X[0,:]
#        return self.X[0:self.c,:]

    def update(self, x, y):
        '''
            Update
            --------------
        '''
        self.X[self.i,:] = x
        self.y[self.i] = y

        #self.error_rate = (y - self.predict(x))**2

        self.i = (self.i + 1) % self.n

        if self.c < self.n:
            self.c = self.c + 1

        self.h.fit(self.X[0:self.c,:], self.y[0:self.c])

def demo():
    pat = array([
       # X X Y
        [9.,0.,0.9],
        [8.,1.,0.8],
        [2.,1.,0.2],
        [1.,0.,0.1],
    ])

    D = 2
    L = 1
    X = array(pat[:,0:D],dtype=float)
    Y = array(pat[:,D:3],dtype=float)
    y = (Y.T)[0].T

    print "#"
    h = kNN(D, L, k=2)
    for i in range(4):
        h.predict(X[i,:].reshape(1,-1))
        h.update(X[i,:],y[i])

    print "============"
    print "X\n", X
    print "y\n", y
    print "yp\n",
    print h.predict(X)
    print "n", h.project()

#demo()
