from numpy import *
from MOP import linear
from MLP import sigmoid
set_printoptions(precision=4)

class STF():

    ''' 
        Recurrent Basis/Transformation Function
        ---------------------------------------
        Turn x into \phi in a recurrent manner.

        Unlike RTF, there is no randomness here. In fact, the output is simply as a leaky recurrent version of each input.
    '''

    W_hh = None
    W_ih = None
    z = None

    def __init__(self, N_i, N_h, f=sigmoid, fade_factor=0.90):
        '''
        '''
        self.f = f              # non-linearity
        self.N_i = N_i          # inputs
        N_h = N_i

        self.z = zeros(N_h+N_i+1)    # nodes
        self.z[0] = 1.

        self.W_hh = fade_factor * eye(N_h,N_h)        # average of each input

    def store_y(self,y):
        print "we can store y (the PREVIOUS output) so as to use it in the transformamtion"

    def phi(self,x):

        self.z[-(self.N_i):] = x[:]
        self.z[1:self.N_i+1] = self.f( dot(self.W_hh, self.z[1:self.N_i+1]) + 0.1 * x )  # <--- wtf is the 0.1 ??
        return self.z

    def reset(self):
        self.z = self.z * 0.
        self.z[0] = 1.

def demo():
    D = 2
    H = 2
    N = 100
    rtf = STF(D,H,f=linear)
    #print rtf.W
    X = ones((N,D)) #(random.rand(N,D) > 0.5) * 1.
    X[:,0] = 0.8  # constant
    X[:,1] = 0.5  # constant
    Z = zeros((N,H*2+1))
    for i in range(N):
        if i > 10:
            X[i,0] = X[i,0] * 0.; #+ i * 0.01
        if i > 20:
            X[i,1] = X[i,1] * 0.; #+ i * 0.01
        #elif i > 10 and i < 20:
        #    X[i,:] = X[i,:] * 0.; #+ i * 0.01
        #elif i > 20 and i < 30:
        #    X[i,:] = X[i,:] * 0.; #+ i * 0.01
        #elif i > 30 and i < 50:
        #    X[i,:] = X[i,:] - i * 0.01
        Z[i] = rtf.phi(X[i])

    print Z

    import matplotlib
    matplotlib.use('Qt4Agg')
    from matplotlib.pyplot import *

    fig = figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0,N])
    ax.set_ylim([-0.1,1.1])
    lines = [None for i in range(H+D+1)]
    for j in range(1):
        #bias
        lines[j], = ax.plot([0,0],"k-",label=r"$z_"+str(j)+"$",linewidth=2)
    for j in range(1,D+1):
        #orig
        lines[j], = ax.plot([0,0],label=r"$z_"+str(j)+"$",linewidth=2)
    for j in range(D+1,D+H+1):
        #hidden
        lines[j], = ax.plot([0,0],"k:",label="x "+str(j),linewidth=3)
    legend()


    ion()
    for lim in range(1,N):
        for j in range(D+H+1):
            lines[j].set_data(range(0,lim),Z[0:lim,j])
        pause(0.1)
    grid(True)
    legend()
    show()
    ioff()

if __name__ == '__main__':
    demo()
