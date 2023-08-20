import numpy as np

def errorFunc(y_ans,y_est,errorType = 3):
    ## Error function to train network
    # Compute error between current estimation and answer from test function
    # errorType // 1: linear, 2: quadratic, 3: sqrt of quadratic
    #
    # Last update: August 19, 2023
    # Author: Jinwook Lee
    #
    
    if errorType == 1:
        error = sum(abs(y_ans-y_est))
    elif errorType == 2:
        error = sum((y_ans-y_est)**2)
    else: # errorType == 3:
        error = np.sqrt(sum((y_ans-y_est)**2))
    return error
