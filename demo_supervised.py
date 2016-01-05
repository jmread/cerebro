from numpy import *
set_printoptions(precision=3, suppress=True)
import matplotlib
#matplotlib.use('GTKAgg')
matplotlib.use('Qt5Agg')
from matplotlib.pyplot import *

'''
'''

# Dataset
N = 100
D = 2
T = N * 100
X = random.randn(N,D)
#w_true = array([1.4, -1.4])
X[:,1] = 1. #NEW
w_true = array([1.5, 1.5]) #NEW
from data import data0
X,y = data0(N,w_true)

def true_fire(x):
    a = dot(x,w_true.T)
    return a
    #return exp(-(1.1 * a)**2)
    #return 1./(1. + exp(-a))

def true_error_density(w_):
    ''' true error '''

    a = dot(X,w_.T)
    p = a
    y = true_fire(X)
    return 1./N * sqrt(sum((y - p)**2))# + 0.001 * sqrt(dot(w_.T,w_))

def plot_surface(ww1,ww2,e_fn):

    z =  zeros((len(ww1),len(ww2)))
    for j in range(len(ww1)):
        w1 = ww1[j]
        for k in range(len(ww2)):
            w2 = ww2[k]
            z[k,j] = e_fn(array([w1,w2]))

    return z

def plot_surface_new(ww1,ww2,e_fn):
    xx, yy = meshgrid(ww1, ww2, sparse=True)
    #z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    #h = plt.contourf(ww1,ww2,z)

    ww = column_stack([xx.flatten(), yy.flatten()])
#    z =  zeros((len(ww1),len(ww2)))
#    for i in range(len(z)):
#            z[k,j] = e_fn(array([w1,w2]))
#    return e_fn(ww)
    #w_ = 
    #a = dot(X,w_.T)
    #p = a
    #y = true_fire(X)
    #z =  1./N * sqrt(sum((y - p)**2)) + 0.001 * sqrt(dot(w_.T,w_))
    #return z


ww1 = arange(-2, 2.5, 0.5)
ww2 = arange(-2, 2.5, 0.5)
history = zeros((10000,2))
c = 0

fig = figure()
ax0 = fig.add_subplot(121)
ax0.set_xlim([-2.0,2.0])
ax0.set_ylim([-2.0,2.0])
ax1 = fig.add_subplot(122)
ax1.set_xlim([-2.0,2.0])
ax1.set_ylim([-2.0,2.0])

from cerebro.functions import linear, sigmoid

b = None
baseline = 1
f_h=linear

if baseline == 2:
    # PS: don't forget to change e to y when learn()ing with this brain!!
    from cerebro.brain import brain_incr
    from cerebro.RLS import RLS, F_sigmoid, F_linear
    from cerebro.MOP import MOPpf
    b = brain_incr(MOPpf(D,1,f=linear),RLS(D*1,1,ridge=10.1,f=linear))
elif baseline == 1:
    from cerebro.brain import make_brain
    b = make_brain(D,0,1,f_h,f_h,density=0.5,tau=1,f_desc="DE")
else:
    from cerebro.brain_q import brain
    #from AC import AC
    from MOP import MOPpf
    b = brain(MOPpf(2,1,f=linear),QL(2,1))

# Plot the true understanding of the world
z = plot_surface(ww1,ww2,true_error_density)
title("true error surface of weights")
ax0.contourf(ww1,ww2,z,50)
ax0.plot(w_true[0],w_true[1],'bx',markersize=5,linewidth=5)

#C1 = ax1.contourf(ww1,ww2,z,50)
L2, = ax1.plot(b.g.get_weight_vector(), 'go', markersize=5) # gol
L3, = ax1.plot(b.g.get_weight_vector(), 'r-', linewidth=2) # history
grid(True)
title("error surface of weights")
ion()
show()

for tt in range(T):
    t = tt % N

    # New sample (input) from the dataset
    x = X[t,:]

    # Fire on input
    p = b.fire(x) 

    # Evaluate the response from the environment
    y = true_fire(x)
    e = sqrt(sum((y - p)**2))# + 0.001 * sqrt(dot(w_.T,w_))

    # Learn from that response
    error = -1
    if baseline == 0:
        error = b.learn(y)
    elif baseline == 1:
        error = b.learn(sigmoid(e))
    else:
        error = b.learn(e)

    print "x=",x, "w=",b.g.wv, "y=", y, "p=", p, "e=", e#, "z=", e_0 #, b.f.w,

    if error >= 0:

        # TODO need to remove first
        # Plot our understanding of the world
        if baseline == 1:
            z = plot_surface(ww1,ww2,b.f.map)
            C1 = None
            C1 = ax1.contourf(ww1,ww2,z,50)
        L2.set_data(w_true[0],w_true[1])
        history[c] = b.g.get_weight_vector()[0:2]
        #print "w[t] =",  b.g.get_weight_vector()
        L3.set_data(history[max(0,c-20):c+1,0],history[max(0,c-20):c+1,1])
        #plot(b.explr.xmin[0],b.explr.xmin[1],'yx',linewidth=5,markersize=5)
        #O = array([b.explr.g.get_weight_vector(),b.explr.g.get_weight_vector()+b.explr.vexp])
        #plot(O[:,0],O[:,1],'y-',linewidth=3)
        c = c + 1
        pause(0.1)


ioff()
