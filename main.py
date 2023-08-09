import random
import numpy as np
import matplotlib.pyplot as plt
from dumb_net import DumbNet
from test_func import testFunc
from error_func import errorFunc

if __name__ == '__main__':
    # Network distribution
    # - Must end with 1 for the last neuron without activation function
    n_layer_dist = np.array([2,5,5,5,1])
    dn = DumbNet(n_layer_dist,initWeight=np.NaN,actFuncType=0)

    # Numerical setup
    eps = 1E-06
    tol = 1E-01
    factor = 1
    N_iter = 300

    # Errors
    eMin_list = []
    eMax_list = []
    eAvg_list = []

    # Iterations
    n_iter_list = range(N_iter)
    for n_iter in n_iter_list:
        """ Immediate update method
        error_list = []
        for x_in in np.arange(-1,1,0.05):
            for n_w in range(dn.N_weight):
                e_init = errorFunc(dn,testFunc,x_in)
                oldWeights = dn.getWeights()

                newWeights = np.copy(oldWeights)
                newWeights[n_w] += eps
                dn.setWeights(newWeights)
                    
                e_updt = errorFunc(dn,testFunc,x_in)
                dedx = (e_updt-e_init)/eps
                
                if abs(dedx)>1E-16:
                    solWeights = np.copy(oldWeights)
                    solWeights[n_w] -= e_init/dedx*factor
                    dn.setWeights(solWeights)
                    
            # Check error
            error_list.append(e_init)
        """

        #""" Update for each x, random order
        error_list = []
        x_in_list = np.arange(-1,1,0.05)
        np.random.shuffle(x_in_list)
        for x_in in x_in_list:
            # Current error
            e_init = errorFunc(dn,testFunc,x_in)
            error_list.append(e_init)
            if abs(e_init)<tol:
                continue

            # Compute Jacobian
            oldWeights = dn.getWeights()
            co_J = np.empty((0,1),dtype=float)
            for n_w in range(dn.N_weight):
                newWeights = np.copy(oldWeights)
                newWeights[n_w] += eps
                dn.setWeights(newWeights)
                    
                e_updt = errorFunc(dn,testFunc,x_in)
                dedx = (e_updt-e_init)/eps
                co_J = np.append(co_J,dedx)
            co_J2D = co_J.reshape(1,len(co_J))
            co_err2D = e_init.reshape(1,1)

            co_b2D = -(co_J2D.transpose()) @ co_err2D
            co_a2D = (co_J2D.transpose()) @ co_J2D
            
            # Update globally
            dx2D = np.linalg.lstsq(co_a2D, co_b2D, rcond=None)[0]
            dx1D = dx2D.reshape(-1,)
            solWeights = oldWeights+dx1D*factor
            dn.setWeights(solWeights)
        #"""

        """ Global update
        oldWeights = dn.getWeights()

        # Compute Jacobian
        x_query_list = np.arange(-1,1,0.05)
        N_x = np.size(x_query_list)
        co_J2D = np.empty((N_x,dn.N_weight),dtype=float)
        co_e2D = np.empty((N_x,1),dtype=float)
        for n_x in range(N_x):
            # Current value
            x_in = x_query_list[n_x]
            e_init = errorFunc(dn,testFunc,x_in)
            co_e2D[n_x][0] = e_init
            for n_w in range(dn.N_weight):
                newWeights = np.copy(oldWeights)
                newWeights[n_w] += eps
                dn.setWeights(newWeights)
                    
                e_updt = errorFunc(dn,testFunc,x_in)
                dedx = (e_updt-e_init)/eps
                co_J2D[n_x][n_w] = dedx
        co_b2D = -(co_J2D.transpose()) @ co_e2D
        co_a2D = (co_J2D.transpose()) @ co_J2D
        
        # Update globally
        dx2D = np.linalg.lstsq(co_a2D, co_b2D, rcond=None)[0]
        dx1D = dx2D.reshape(-1,)
        solWeights = oldWeights+dx1D*factor
        dn.setWeights(solWeights)
        error_list = co_e2D.reshape(-1,)
        """

        # Iteration status
        eMin_list.append(min(error_list))
        eMax_list.append(max(error_list))   
        eAvg_list.append(sum(error_list)/len(error_list))        
        print("Iteration {}, minE={:.2e}, maxE={:.2e}, avgE={:.2e}".format(
               n_iter,min(error_list),max(error_list),sum(error_list)/len(error_list)))
    
    # Status
    print(" ")
    print("===")
    for n_layer in range(dn.N_layer):
        for n_width in range(dn.N_width[n_layer]):
            curWeights = dn.neurons[n_layer][n_width].getWeights()
            curStr = str(curWeights)
            print("Neuron {}-{}".format(n_layer,n_width) + ": " + curStr)
    print(" ")
        
    ## Plot
    # Convergence
    fig, ax = plt.subplots()
    ax.semilogy(n_iter_list,eMin_list,label='Error Min')
    ax.semilogy(n_iter_list,eMax_list,label='Error Max')
    ax.semilogy(n_iter_list,eAvg_list,label='Error Avg')
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
