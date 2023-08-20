import numpy as np
import matplotlib.pyplot as plt
from dumb_net import DumbNet
from test_func import testFunc
from error_func import errorFunc

## Objective and procedure
# Train network that takes single input for single output
# Update default functionType in test_func.py to aim for different targets
# Adjust Setup class parameters for network tuning

class Setup():
    ## Class to control setup
    def __init__(self):
        # Basics&
        self.eps = 1E-06 # infinitesimal perturbation to compute gradient
        self.conv = 0.1 # Convergence threshold
        self.tol = self.conv*0.3 # Acceptable range of error to skip updating
        self.N_iter = 3000 # Iteration count
        self.N_trial = 5 # Trial attempt
        self.weightTrainFrac = 0.3 # Fraction of weight to train per each iteration
        self.xFocusFrac = 0.3 # Fraction of test x to focus on per each iteration
        self.weightLearnRate = 0.1  # Weight learning rate

        # Train range
        self.x_train_min = -1
        self.x_train_max = +1
        self.N_x_train = 51

        # Test range
        self.x_test_min = -1
        self.x_test_max = +1
        self.N_x_test = 501

        # Layer distribution
        # - Must end with 1 for the last neuron without activation function
        self.n_layer_dist = np.array([9,9,9,1])

if __name__ == '__main__':
    ## Initialize
    input = Setup()

    ## Train network
    for n_trial in range(input.N_trial):
        print("\n=== Attempt #{:d} ===".format(n_trial))
        dn = DumbNet(input.n_layer_dist)
        eMinMaxAvg_list = dn.train(input,testFunc,errorFunc)
        if dn.isTrained:
            break
    
    ## Review
    if not dn.isTrained:
        print("Training failed")
    else:
        # Weights
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
        n_iter_list = np.arange(len(eMinMaxAvg_list[0]))
        fig, ax = plt.subplots()
        ax.semilogy(n_iter_list,eMinMaxAvg_list[0],label='Error Min')
        ax.semilogy(n_iter_list,eMinMaxAvg_list[1],label='Error Max')
        ax.semilogy(n_iter_list,eMinMaxAvg_list[2],label='Error Avg')
        ax.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(axis='both', color='0.95')

        # Test query
        x_test = np.linspace(input.x_test_min,input.x_test_max,input.N_x_test)
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
