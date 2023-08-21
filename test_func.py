import numpy as np
def testFunc(x_in,N_order_x,N_order_y,functionType = 1):
    ## Target test function to which the network is trained
    # N_order_x == 1 and N_order_y == 1:
    #    1: Linear
    #    2: Sine wave
    #    3: Quadratic
    #    4: Absolute
    #    5: Square wave
    #    6: Piecewise linear
    #
    # N_order_x == 2 and N_order_y == 2:
    #    1: Linear-Linear
    #
    # N_order_x == 2 and N_order_y == 1:
    #    1: Ellipsoid
    #
    # Last update: August 21, 2023
    # Author: Jinwook Lee
    #

    # Definition of each function
    if N_order_x == 1 and N_order_y == 1:
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
            
    elif N_order_x == 2 and N_order_y == 2:
        if functionType == 1:
            x_re = np.reshape(x_in,(N_order_x,-1))
            y_re = np.zeros(np.shape(x_re))
            y_re[0,:] = +2*x_re[0,:] - 1
            y_re[1,:] = -1*x_re[1,:] + 2
            y_out = np.reshape(y_re,np.shape(x_in))
            
    elif N_order_x == 2 and N_order_y == 1:
        if functionType == 1:
            y_temp = 1-(x_in[0]**2+x_in[1]**2)
            if np.size(y_temp)==1:
                if y_temp<0:
                    y_temp = 0
            else:
                i_sel = y_temp<0
                y_temp[i_sel] = 0
            y_out = np.sqrt(y_temp)

    else:
        print("Invalid order type")
        y_out = np.NaN(np.size(x_in))

    return y_out