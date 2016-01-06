from numpy import *
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.pyplot import *
set_printoptions(precision=3, suppress=True)

'''
'''

WIDTH=200
DIAG=sqrt((WIDTH*2)**2 + (WIDTH*2)**2)

def rwd(p1,p2):
    return 1. - (sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) / DIAG)

# WORLD STATE
gol = random.randn(2) * WIDTH/4
pos = random.randn(2) * WIDTH/4

# Initial reward
r = rwd(gol,pos)

# BRAIN STATE
from cerebro.brain import make_brain
from cerebro.functions import linear
#b = make_brain(2,0,2,f_desc='QL',f_o=linear)
b = make_brain(2,10,2,f_desc='DE2',f_o=linear,tau=10)   

# (Setup Animation)
T = 10000

history = zeros((T,2))
history[0,:] = pos
fig = figure(figsize=(16.0, 10.0))
from matplotlib import gridspec
gs = gridspec.GridSpec(2, 1, height_ratios=[1,4])
ax_1 = fig.add_subplot(gs[0])
ax_1.set_ylim([-10,10])
ax_1.set_title('weights and reward')
Lk, = ax_1.plot([0,0],[0,0], 'k-', linewidth=3) 
Lw, = ax_1.plot(range(len(b.g.wv)),zeros(len(b.g.wv)), 'mo-', linewidth=1) 
ax_2 = fig.add_subplot(gs[1])
ax_2.set_xlim([-WIDTH,WIDTH])
ax_2.set_ylim([-WIDTH,WIDTH])
Ln, = ax_2.plot(pos, 'ro', markersize=5) # pos
L2, = ax_2.plot(gol, 'go', markersize=5) # gol
L3, = ax_2.plot(pos, 'r-', linewidth=2) # history
gs.tight_layout(fig,h_pad=0.1)
ion()
grid(True)
show()
e0 = random.rand()
for t in range(1,T):

    # == Update Environment
    theta = 0.01
    R = array([[cos(theta), -sin(theta)],[sin(theta),cos(theta)]])
    gol = dot(R,gol)
    #gol[1] = gol[1] + 0.05
    L2.set_data(gol)

    # == (Plot)
    #print b.W
    #print r, b.i, b.w[b.i]

    # == Observation / SET INPUT (note: no bias at the moment!)
    x = gol - pos
    print "x", x

    # == Act / PROCESS THE INPUT
    y = b.fire(x) 
    pos = pos + y 
    #pos = clip(pos + y[0:2],[0,0],[WIDTH,WIDTH])
    history[t,:] = pos
    L3.set_xdata(history[0:t+1,0])
    L3.set_ydata(history[0:t+1,1])
    #plot(history[0:t+1,0],history[0:t+1,1],"r-",markersize=5)
    #plot(pos[0],pos[1],'ro',markersize=5,linewidth=5)
    Ln.set_xdata(pos[0])
    Ln.set_ydata(pos[1])

    # Reward @TODO should be same as observation 'x'
    r = rwd(gol,pos)
    e = 1.-r # (convert reward to error)
    #if e < e0:
    #    fe = 0.
    #else:
    #    fe = e
    #print e,e0,fe
    #e = 0.1 * e + 0.9 * e * clip((e - e0) + 1.,0.,1.)
    #print e
    Lk.set_xdata([0,r*10.])
    Lw.set_ydata(b.g.wv)

    # == PROCESS INTERNALS
    b.learn(e)       
    e0 = e

    # == Draw
    #g.canvas.draw()
    #fig.clf() 
    pause(0.01)

    if pos[0] > WIDTH or pos[0] < -WIDTH:
        pos = pos * 0.
    if pos[1] > WIDTH or pos[1] < -WIDTH:
        pos = pos * 0.

ioff()
