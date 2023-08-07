import numpy as np

def testFunc(x_in):
    #return x_in*10
    #return np.sin(x_in)
    #return x_in**2
    #return 1 if x_in>0 else -1
    #return abs(x_in)

    x_in = np.array(x_in,dtype=float)
    i_sel = x_in>0
    y_out = np.array(x_in*0.25,dtype=float)
    y_out[~i_sel] = 0
    return y_out