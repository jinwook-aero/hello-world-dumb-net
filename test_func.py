import numpy as np

def testFunc(x_in):
    #return x_in*2-1
    return np.sin(x_in*np.pi*1.5)
    #return x_in**2
    #return abs(x_in)

    """
    x_in = np.array(x_in,dtype=float)
    i_sel0 = (x_in>=-1) & (x_in<-0.5)
    i_sel1 = (x_in>=-0.5) & (x_in<0)
    i_sel2 = (x_in>=0) & (x_in<0.5)
    i_sel3 = (x_in>=0.5) & (x_in<=1)
    y_out = np.zeros(np.size(x_in),dtype=float)
    y_out[i_sel0] = -1
    y_out[i_sel1] = +1
    y_out[i_sel2] = -1
    y_out[i_sel3] = +1
    return y_out
    """

    """
    x_in = np.array(x_in,dtype=float)
    i_sel = x_in>0.5
    y_out = np.array(x_in*-1,dtype=float)
    y_out[~i_sel] = 1
    return y_out
    """
