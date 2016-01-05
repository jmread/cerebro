from numpy import *
set_printoptions(precision=3)

def data0(N, w_true=array([1.5, 0.0])):
    '''
        y[t] = 0.5 x[t]
    '''

    X = random.randn(N,2)
    X[:,1] = 1. #NEW
    y = dot(X,w_true.T)
    return X,y

def data1(N, w_true=array([2.5, 0.0])):
    '''
        y[t] = 0.5 x[t-1]
    '''

    X_seq = random.rand(N,2)
    X_seq[:,1] = 1.
    y_seq = zeros((N,1))

    for i in range(1,N):
        y_seq[i] = dot(X_seq[i-1],w_true.T)

    return X_seq, y_seq


from cerebro.functions import sigmoid
def data1b(N):
    '''
        X[t] = data1(t)
        y[t] = data1(t) - 0.3
    '''
    X_seq, y_seq = data1(N)
    y_seq = -0.3 + y_seq
    return X_seq, y_seq



def data_url0():
    ''' 
    X: [0 0] is a space, [1 1] is a character, [0 1] is a stop 
    Y: 1 = part of valid URL  (the case for characters and spaces)
    '''

    X_seq = asarray(matrix('0 0; 1 1; 1 1; 1 1; 0 1; 1 1; 1 1; 0 1; 0 0;     1 1; 1 1; 1 1; 0 1; 1 1; 1 1; 1 1; 0 0'))
    y_seq = asarray(matrix('0 1 1 1 1 1 1 0 0   1 1 1 1 1 1 1 1 0').T)
    return X_seq, y_seq

def data_url2():
     ''' 
        Task: Mark the end of each URL
     '''
     sx=" x xxxx xx xxx.xxxxxx.xx xxx xxx.xxxxxxx.xxx. xxxx, x xxxxxxxxxx xxxx xxxx xxxx://xxxx.xx.xxx xxxx xxx xxxxxxxxx: xxxxx.xxx,xxxxxx.xxx,xxxx://xxx.xxxxx.xxxxxx.xxx. "
     sy="00000000000000000000000010000000000000000000010000000000000000000000000000000000000000000000010000000000000000000000000000010000000000100000000000000000000000000010"
     y_seq = [int(x) for x in list(sy)]
     N = len(y_seq)
     X_seq = zeros((N,6))
     dic = {
             ' ' : array([1,0,0,0,0,0]),
             'x' : array([0,1,0,0,0,0]),
             '.' : array([0,0,1,0,0,0]),
             '/' : array([0,0,0,1,0,0]),
             ':' : array([0,0,0,0,1,0]),
             ',' : array([0,0,0,0,0,1]),
     }
     c = 0
     for x in list(sx):
         X_seq[c,:] = dic[x]
         c = c + 1
         
     return X_seq, y_seq


def data_text():
     ''' 
        Task: Mark the end of each word with the number of characters it contains.
     '''
     sx="x xxxx xx xxxxx xxx xxx xxxx "
     sy="01000040020000050003000300004"
     #sy="01000010010000010001000100001"
     y_seq = [int(x) for x in list(sy)]
     N = len(y_seq)
     D = 2
     X_seq = zeros((N,D))
     dic = {
             ' ' : array([1,0]),
             'x' : array([0,1]),
     }
     c = 0
     for x in list(sx):
         X_seq[c,:] = dic[x]
         c = c + 1
         
     return X_seq, y_seq, sx

def data_url3():
     ''' 
        Task: Mark the end of each URL with the number of characters it contains.
     '''
     sx=" x xxxx xx xx.x.xx xxx xxx.xxx. xxxx, x xxxxxxxxxx xxxx. xxxx, xxx: xxx.xx,xxxxx.xxx, xxx. "
     sy="0000000000000000007000000000007000000000000000000000000000000000000000000060000000009000000"
     y_seq = [int(x) for x in list(sy)]
     N = len(y_seq)
     X_seq = zeros((N,6))
     dic = {
             ' ' : array([1,0,0,0,0,0]),
             'x' : array([0,1,0,0,0,0]),
             '.' : array([0,0,1,0,0,0]),
             '/' : array([0,0,0,1,0,0]),
             ':' : array([0,0,0,0,1,0]),
             ',' : array([0,0,0,0,0,1]),
     }
     c = 0
     for x in list(sx):
         X_seq[c,:] = dic[x]
         c = c + 1
         
     return X_seq, y_seq, sx



if __name__ == '__main__':
    print data_url2()
    
