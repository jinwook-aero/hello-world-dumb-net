import numpy as np

def errorFunc(dn,testFunc,x_in):
    y_est = dn.in2out(x_in)
    y_ans = testFunc(x_in)
    #return sum(y_ans-y_est)
    return sum((y_ans-y_est)**2)
