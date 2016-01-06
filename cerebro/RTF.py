from numpy import *
from functions import sigmoid
set_printoptions(precision=4)

class RTF():

    ''' 
        Recurrent Basis/Transformation Function
        ---------------------------------------
        Turn x into \phi in a recurrent manner.
    '''

    W_hh = None
    W_ih = None
    z = None

    def __init__(self, N_i, N_h, f=sigmoid, density=0.1):
        '''
        '''
        self.f = f              # non-linearity
        self.N_i = N_i          # inputs

        # Generate nodes
        self.z = zeros(N_h)    # nodes
        self.z[0] = 1.        # output bias node

        # Generate random weights
        self.W_ih = random.randn(N_i,N_h-1) * 1.0 * (random.rand(N_i,N_h-1) <= density)
        self.W_hh = random.randn(N_h-1,N_h-1) * 1.0 * (random.rand(N_h-1,N_h-1) <= density)

        # Calculate the eigenvectors (V) of W_hh
        V,U = linalg.eig(self.W_hh)
        # Check that we won't be dividing by 0
        if max(absolute(V)) <= 0.:
            V = V + 0.01
        # Scale the initial weights to a spectral radius of 1.
        self.W_hh = self.W_hh / max(absolute(V))

        #self.b_ih = random.randn(N_h-1) * 0.1

    def store_y(self,y):
        print "we can store y (the PREVIOUS output) so as to use it in the transformamtion"

    def phi(self,x):
        #print "+++++++++++"
        #print self.W_hh.shape
        #print self.W_ih.shape
        ##print self.b_ih.shape
        #print x.shape
        #print self.z.shape
        #print "==========="
        self.z[1:] = self.f( dot(self.W_hh, self.z[1:]) + dot(self.W_ih.T, x) ) #self.b_ih +         <--- I don't think bias is needed for ESN??
        return self.z

    def reset(self):
        self.z = self.z * 0.
        self.z[0] = 1.

class RTFv2(RTF):

    ''' 
        Like RTF, but includes (@TODO)

            - output feedback loop
            - regularization (noise to the input)
            - efficient sparse solution (each node is connected to exactly N other nodes) -- similary to Markov Chain code for Jaakko's seminar course.

    '''

    W_oh = None
    y = None
    v = None

    def __init__(self, N_i, N_h, N_o, f=sigmoid, density=0.1, state_noise=0.01):
        RTF.__init__(self,N_i,N_h,f,density)
        self.N_o = N_o          # outputs
        self.W_oh = random.randn(N_o,N_h-1) * 1.0 * (random.rand(N_o,N_h-1) <= density)      # NEW
        self.v = state_noise

    def store_y(self,y):
        self.y = y

    def phi(self,x):

        self.z[0:-1] = self.f( dot(self.W_hh, self.z[0:-1]) + dot(self.W_ih.T, x) + dot(self.W_oh.T, self.y)) + random.randn(len(self.z)-1) * self.v 
        return self.z


def demo():
    D = 2
    H = 10
    N = 100
    rtf = RTF(D,H,f=sigmoid,density=0.2)
    #print rtf.W
    X = random.randn(N,D) #(random.rand(N,D) > 0.5) * 1.
    X[:,0] = 1.
    X[10:20,:] = 0.
    X[40:60,:] = 0.
    X[80:100,:] = 0.
    Z = zeros((N,H))
    for i in range(N):
        Z[i] = rtf.phi(X[i])

    import matplotlib
    matplotlib.use('Qt4Agg')
    from matplotlib.pyplot import *

    fig = figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0,N])
    ax.set_ylim([-0.1,1.1])
    lines = [None for i in range(H+D)]
    for j in range(D):
        lines[j], = ax.plot([0,0],"k:",label=""+str(j),linewidth=2)
    for j in range(D,H+D):
        lines[j], = ax.plot([0,0],label=""+str(j),linewidth=2)
    ion()
    for lim in range(1,N):
        for j in range(D):
            lines[j].set_data(range(0,lim),X[0:lim,j])
        for j in range(H):
            lines[j].set_data(range(0,lim),Z[0:lim,j])
        pause(0.1)
    grid(True)
    legend()
    show()
    ioff()

if __name__ == '__main__':
    demo()
