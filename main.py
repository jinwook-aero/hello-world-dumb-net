import numpy as np
import matplotlib.pyplot as plt
from dumb_net import DumbNet
from test_func import testFunc

if __name__ == '__main__':
    # Run the case
    n_layer_dist = np.array([1, 3, 1])
    #n_layer_dist = np.array([1, 5, 5, 5, 5, 5, 1])
    dn = DumbNet(n_layer_dist,initWeight=1,actFuncType=0)

    # Test function
    x_in = 1
    y_ans = testFunc(x_in)
    y_cur = dn.in2out(x_in)
    print("y_ans = {}".format(y_ans))
    print("y_cur = {}".format(y_cur))

    # Backpropagation
    N_layer = dn.N_layer
    N_width = dn.N_width
    N_weight = dn.N_weight
    eps = 1E-06
    factor = 1/N_weight

    N_iter = N_weight*5
    n_iter_list = range(N_iter)
    eMin_list = []
    eMax_list = []
    for n_iter in n_iter_list:
        error_list = []
        for x_in in np.arange(0.1,1,0.1):
            # Current value
            y_ans  = testFunc(x_in)
            y_init = dn.in2out(x_in)
            oldWeights = dn.getWeights()

            # Compute Jacobian
            N_w = dn.N_weight
            co_J = []
            for n_w in range(N_w):
                newWeights = np.copy(oldWeights)
                newWeights[n_w] += eps
                dn.setWeights(newWeights)
                
                y_pert = dn.in2out(x_in)
                dydx = (y_pert-y_init)/eps
                co_J.append(dydx)
            co_J1D = np.array(co_J)
            y_err = np.array(y_ans - y_init)

            co_J2D = co_J1D.reshape(1,N_w)
            co_yerr2D = y_err.reshape(1,1)

            co_b2D = (co_J2D.transpose()) @ co_yerr2D
            co_a2D = (co_J2D.transpose()) @ co_J2D
            
            # Update globally
            dx2D = np.linalg.lstsq(co_a2D, co_b2D, rcond=None)[0]
            dx1D = dx2D.reshape(N_w,)
            solWeights = oldWeights+dx1D*factor
            dn.setWeights(solWeights)

            # Check error
            error_list.append(abs(y_err))

        # Iteration status
        eMin_list.append(min(error_list))
        eMax_list.append(max(error_list))
        print("Iteration {}, minE={:.2e}, maxE={:.2e}, minW={:.2f}, maxW={:.2f}".format(
               n_iter,min(error_list),max(error_list),min(oldWeights),max(oldWeights)))

    # Status
    print(" ")
    print("===")
    for n_layer in range(N_layer):
        for n_width in range(N_width[n_layer]):
            curStr = "["
            for curWeight in dn.neurons[n_layer][n_width].getWeights():
                curStr += "{:.3e}".format(curWeight) + ", "
            curStr += "]"
            curStr = curStr.replace(", ]", "]")
            print("Neuron {}-{}".format(n_layer,n_width) + ": " + curStr)
    print(" ")
        
    ## Plot
    # Convergence
    fig, ax = plt.subplots()
    ax.semilogy(n_iter_list,eMin_list,label='Error Min')
    ax.semilogy(n_iter_list,eMax_list,label='Error Max')
    ax.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(axis='both', color='0.95')
    #plt.show()

    # Test query
    x_test = np.linspace(-1,1,101)
    y_answer = testFunc(x_test)
    y_model = []
    for x_in in x_test:
        y_model.append(dn.in2out(x_in))
    y_model = np.array(y_model)

    fig, ax = plt.subplots()
    ax.plot(x_test,y_answer,label='Answer')
    ax.plot(x_test,y_model,label='Model')
    ax.legend()
    plt.xlabel('X'); plt.ylabel('Y')
    plt.grid(axis='both',color='0.95')
    plt.show()
