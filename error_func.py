import numpy as np

def errorFunc(dn,testFunc,x_in,errorType = 3):
    ## Error function to train network
    # Compute error between current estimation and answer from test function
    # errorType // 1: linear, 2: quadratic, 3: sqrt of quadratic
    
    y_est = dn.in2out(x_in)
    y_ans = testFunc(x_in)
    if errorType == 1:
        error = sum(abs(y_ans-y_est))
    elif errorType == 2:
        error = sum((y_ans-y_est)**2)
    else: # errorType == 3:
        error = np.sqrt(sum((y_ans-y_est)**2))
    return error
