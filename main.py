import numpy as np
import matplotlib.pyplot as plt
import time
from dumb_net import DumbNet
from test_func import testFunc
from error_func import errorFunc
import cProfile
import subprocess

## Objective and procedure
# Train network that takes single input for single output
# Update default functionType in test_func.py to aim for different targets
# Adjust Setup class parameters for network tuning
#
# Last update: August 19, 2023
# Author: Jinwook Lee
#

class Setup():
    ## Class to control setup
    def __init__(self):
        # Basics
        self.eps = 1E-06 # infinitesimal perturbation to compute gradient
        self.conv = 0.1 # Convergence threshold
        self.tol = self.conv*0.3 # Acceptable range of error to skip updating
        self.N_iter = 50000 # Iteration count
        self.N_trial = 5 # Trial attempt
        self.weightTrainFrac = 0.3 # Fraction of weight to train per each iteration
        self.weightLearnRate = 0.1  # Weight learning rate

        # Train set
        self.x_train_min = -1
        self.x_train_max = +1
        self.N_x_train = 51

        # Test set
        self.x_test_min = -1
        self.x_test_max = +1
        self.N_x_test = 501

        # Layer distribution
        # - Must end with 1 for the last neuron without activation function
        self.n_layer_dist = np.array([10,10,10,1])

def main():
    ## Initialize
    t_start = time.perf_counter()
    input = Setup()

    ## Train network
    for n_trial in range(input.N_trial):
        print("\n=== Attempt #{:d} ===".format(n_trial))
        dn = DumbNet(input.n_layer_dist)
        eMinMaxAvg_list = dn.train(input,testFunc,errorFunc)
        if dn.isTrained:
            break
        
    ## Duration
    t_end = time.perf_counter()
    print("\nElapsed time {:.5f} seconds\n".format(t_end-t_start))
    
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

if __name__ == '__main__':
    ## Profiling main
    profileName = 'temp.prof'
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats(profileName)
    
    snakeVizCmd = "python -m snakeviz " + profileName
    removeCmd_dos = "del " + profileName
    removeCmd_linux = "rm " + profileName
    p = subprocess.Popen(snakeVizCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    userInput = input("Enter to complete")
    p = subprocess.Popen(removeCmd_dos, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    p = subprocess.Popen(removeCmd_linux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)