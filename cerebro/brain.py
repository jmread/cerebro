from numpy import *
set_printoptions(precision=5)

from functions import linear

def make_brain(D,H,L,f_o=linear,f_h=tanh,f_desc="DE",use_bias=False,density=0.5,tau=1,threshold=10,f_sum=mean,alpha=0.1,ridge=10.1,reg=0.0):

    p_basisfn = None
    bias_index = 0
    if H > 0:
        from RTF import RTF
        p_basisfn = RTF(D,H,f=f_h,density=density)
    elif H < 0:
        from STF import STF
        p_basisfn = STF(D,D*2+1,f=f_h,fade_factor=0.5)
        H = D * 2 + 1
        bias_index = 1
    else:
        H = D

    from MOP import MOPpf
    g = MOPpf(H,L,f=f_o,use_bias=use_bias)

    f_learner = None
    if f_desc == "DE":
        from DE import DE
        # for basic supervised problem (mad_0.py)
        f_learner = DE(len(g.wv),1,threshold,f=linear)
    elif f_desc == "DEnn":
        from DE import DE
        # for basic supervised problem (mad_0.py)
        f_learner = DE(len(g.wv),1,threshold,f=linear,h_desc="MLP")
    elif f_desc == "DE2":
        from DE import DE, rwd
        # for mad_sci.py, takes into account moving terrain
        f_learner = DE(len(g.wv),1,threshold,f_rwd=rwd)
    elif f_desc == "DR":
        from DE import DEr
        f_learner = DEr(len(g.wv),1,g.N_i,threshold)
    elif f_desc == "QL":
        import QL
        return QL.make_brain(D,H,L,f_o)
    elif f_desc == "BPR":
        from BPR import BPR
        f_learner = BPR(len(g.wv),1,f=f_o,alpha=alpha,reg=reg,use_bias=bias_index)
        if g.f != f_learner.f:
            print "WARNING! - FUNCTIONS DO NOT MATCH"
            exit(1)
    elif f_desc == "RLS":
        from RLS import RLS, F_sigmoid, F_linear
        f_learner = RLS(len(g.wv),1,ridge=ridge,f=f_o)
        if g.f != f_learner.f:
            print "WARNING! - FUNCTIONS DO NOT MATCH"
            exit(1)
    else:
        print "BAD SPECIFICATION"
        exit(1)

    if tau <= 1:
        return brain_incr(g,f_learner,p_basisfn)

    else:
        return brain_batch(g,f_learner,p_basisfn,tau=tau,f_sum=mean)

class brain():
    '''
        TODO: the other two classes should extend this class!
    '''
    def copy_of(self):
        print "must implement this"


class brain_incr():

    def copy_of(self):
        from copy import deepcopy
        g = deepcopy(self.g)
        f = deepcopy(self.f)
        p = deepcopy(self.p)
        return brain_incr(g,f,p,tau=clip(int(self.tau + random.randn()),5,1000))

    '''
        Brain
        -----

        It has the two importance interfaces of a brain: learning and responding (firing).

    '''

    def __test(self):
        ''' it is fundamental that these two weight vectors are linked! '''
        self.f.w[0] = 0.1
        self.g.wv[1] = 0.2
        print self.f.w, self.g.wv

    def __to_string(self):
        s = "==========================\n"
        #s = s + g.__to_string(self)
        #s = s + f.__to_string(self)
        #s = s + "==========================\n"
        return s

    def __init__(self, g, f, p=None):
        '''
            g:  firer
            f:  learner
            p:  basis fn
        '''

        if len(g.wv) != len(f.w):
            print "MAJOR ERROR - VECTORS DO NOT MATCH"
            exit(-1)
        elif g.f != f.f:
            print "WARNING! - FUNCTIONS DO NOT MATCH"

        self.g = g
        self.f = f
        self.p = p

        self.g.wv = self.f.w
        self.x = zeros(self.g.N_i)
        #print self.__test()

    def learn(self,y):
        if y is None or self.x is None:
            print "Error: you didn't call fire first!"
            return -1
        self.update(self.x,y)
        self.g.wv[:] = self.f.w[:]
        return y

    def update(self,x,y):
        # NOTE: should automatically set self.g.wv (and respective weights) also
        w = self.f.step((x,y))
        self.g.set_weight_vector(w)
        #w = self.f.step((x,y))
        #self.g.move(v_exp=w,reset=True)

    def fire(self,x):
        if self.p is not None:
            x = self.p.phi(x)     # basis fn
        self.x = x                # store sample
        return self.g.predict(x[:])  # we just got environmental input x, reflex response to that!
        
        


def error(record):
    '''
        error function / inverse reward function
        ----------------------------------------
        record is a record of the error (i.e., inverse reward) over the time window
        use it to calculate a reward!
    '''
    #return mean(record)                                         # the average
    return mean(record) + (record[-1] - record[0])      # the average plus the difference
    #return (record[-1] - record[0])      # the difference

def error_1(record):
    '''
        error function / inverse reward function
        ----------------------------------------
        record is a record of the error (i.e., inverse reward) over the time window
        use it to calculate a reward!
    '''
    return mean(record)                                         # the average
    #return mean(record) + (record[-1] - record[0])      # the average plus the difference
    #return (record[-1] - record[0])      # the difference

class brain_batch():

    def copy_of(self):
        from copy import deepcopy
        g = deepcopy(self.g)
        f = deepcopy(self.f)
        p = deepcopy(self.p)
        return brain_batch(g,f,p,tau=clip(int(self.tau + random.randn()),5,1000),f_sum=deepcopy(self.f_sum))

    '''
        Brain
        -----

        * Learning
            1. store output (y, or e)
            2. when t = tau, use to update f
               (possibly summarizing)

        * Firing
            1. store input (x)
            2. fire back response (from g)

        It has the two importance interfaces of a brain: learning and responding (firing).

    '''

    def __init__(self, g, f, p=None, tau=25, f_sum=None):
        '''
            Initialize
            -----------
            
            f_sum: (summarization function) If to be applied to reinforcement learning, then we need some f_sum to summarize the reward/error, e.g., the mean, the end - beginning, etc.
            (the case where the output is the error)
        '''

        self.g = g
        self.f = f
        self.p = p

        self.t = 0
        self.tau = tau
        self.e_his = zeros(tau)
        self.f_sum = f_sum

        self.g.wv = self.f.w
        self.X = zeros((self.tau,self.g.N_i))

    def learn(self,e):
        ''' 
            LEARN from (x[t-tau],...,x[t], e[t])
            ----------------
            * We just got error 'e' (i.e., inverse reward), 
            * Learn from that to improve master weights W (via dense weights in this case) 
        '''
        self.e_his[self.t] = e                  # record error e[t]
        self.t = self.t + 1                     # next timestep
        if self.t == self.tau:                  # if we are at the end of the window
            err = self.f_sum(self.e_his)
            self.update(self.X[0:self.t],err)       # learn
            self.t = 0                              # reset timestep
            #if self.p is not None:
            #    self.p.reset()                      # <-- be careful doing this !! it is not needed for ALife at all anyway!
            return err                              # return the error (for accounting purposes)

        return -1

    def update(self,sampleX,e_true):
        '''
            Update
            -------
            the sampleX are the values which lead to e_true happening
        '''
        #e_true = e_true + 0.1 * self.g.weight_penalty()
        w = self.f.step((sampleX,e_true))
        self.g.set_weight_vector(w)

        if not self.g.check_():
            print "DID NOT WORK!"
            print self.g.wv, self.g.W
            exit(1)
        #self.g.move(v_exp=w,X=sampleX,reset=True)

    def fire(self,x):
        ''' basis fn '''
        if self.p is not None:
            x = self.p.phi(x)       # basis fn

        ''' store sample '''
        self.X[self.t,:] = x

        ''' we just got environmental input x, reflex response to that '''
        return self.g.predict(x)
        
