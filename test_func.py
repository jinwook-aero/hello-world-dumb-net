import numpy as np
def testFunc(x_in,functionType = 6):
    ## Target test function to which the network is trained
    # functionType
    # 1: Linear
    # 2: Sine wave
    # 3: Quadratic
    # 4: Absolute
    # 5: Square wave
    # 6: Piecewise linear
    #
    # Last update: August 19, 2023
    # Author: Jinwook Lee
    #

    # Definition of each function
    if functionType == 1:
        y_out = x_in*2-1

    elif functionType == 2:
        y_out = np.sin(x_in*np.pi)

    elif functionType == 3:
        y_out = x_in**2

    elif functionType == 4:
        y_out = abs(x_in)

    elif functionType == 5:
        x_in = np.array(x_in,dtype=float)
        i_sel0 = (x_in>=-1) & (x_in<-0.5)
        i_sel1 = (x_in>=-0.5) & (x_in<0)
        i_sel2 = (x_in>=0) & (x_in<0.5)
        i_sel3 = (x_in>=0.5) & (x_in<=1)
        y_out = np.zeros(np.size(x_in),dtype=float)
        y_out[i_sel0] = -1
        y_out[i_sel1] = +1
        y_out[i_sel2] = -1
        y_out[i_sel3] = 0

    elif functionType == 6:
        x_in = np.array(x_in,dtype=float)
        i_sel = x_in>0.5
        y_out = np.array(x_in*-1,dtype=float)
        y_out[~i_sel] = 1

    return y_out