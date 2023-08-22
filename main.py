import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from dumb_net import DumbNet
from test_func import testFunc
from error_func import errorFunc
import cProfile
import subprocess

## Objective and procedure
# Train network that takes vector input (x) for vector output (y)
# Update default functionType in test_func.py to aim for different targets
# Adjust Setup class parameters for network tuning
#
# Last update: August 21, 2023
# Author: Jinwook Lee
#

class Setup():
    ## Class to control setup
    def __init__(self):
        # Basics
        self.eps = 1E-06 # infinitesimal perturbation to compute gradient
        self.conv = 0.25 # Convergence threshold
        self.tol = self.conv*0.3 # Acceptable range of error to skip updating
        self.N_iter = 100_000 # Iteration count
        self.N_trial = 5 # Trial attempt
        self.weightTrainFrac = 0.2 # Fraction of weight to train per each iteration
        self.weightLearnRate = 0.1  # Weight learning rate

        # Network In/Out dimension
        self.N_order_x = 2
        self.N_order_y = 1
        
        # Train set
        self.x_train_min = np.zeros(self.N_order_x)
        self.x_train_max = np.zeros(self.N_order_x)
        self.N_x_train = 51
        for n_order_x in range(self.N_order_x):
            self.x_train_min[n_order_x] = -1
            self.x_train_max[n_order_x] = +1

        # Test set
        self.x_test_min = np.zeros(self.N_order_x)
        self.x_test_max = np.zeros(self.N_order_x)
        self.N_x_test = 501
        for n_order_x in range(self.N_order_x):
            self.x_test_min[n_order_x] = -1
            self.x_test_max[n_order_x] = +1

        # Layer distribution
        # - First and last widths determine
        #   the vector dimension of input and output, respectively
        self.n_layer_dist = np.array([self.N_order_x,10,10,10,10,10,self.N_order_y])

def main(profileName):
    ## Initialize
    input = Setup()
    profiler = cProfile.Profile()
    profiler.enable()

    ## Train network
    for n_trial in range(input.N_trial):
        print("\n=== Attempt #{:d} ===".format(n_trial))
        dn = DumbNet(input.n_layer_dist)
        eMinMaxAvg_list = dn.train(input,testFunc,errorFunc)
        if dn.isTrained:
            break
        
    ## Dump profiling
    profiler.disable()
    profiler.dump_stats(profileName)
    
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
        i_sel = np.invert(np.isnan(eMinMaxAvg_list[0]))

        fig, ax = plt.subplots()
        ax.semilogy(n_iter_list[i_sel],eMinMaxAvg_list[0][i_sel],marker='o',label='Error Min')
        ax.semilogy(n_iter_list[i_sel],eMinMaxAvg_list[1][i_sel],marker='o',label='Error Max')
        ax.semilogy(n_iter_list[i_sel],eMinMaxAvg_list[2][i_sel],marker='o',label='Error Avg')
        ax.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(axis='both', color='0.95')

        # Test query
        if input.N_order_x == 1 and input.N_order_y == 1:
            x_test = np.linspace(input.x_test_min[0],input.x_test_max[0],input.N_x_test)
            y_answer = testFunc(x_test,input.N_order_x,input.N_order_y)
            y_model = []
            for x_in in x_test:
                y_model.append(dn.in2out([x_in])[0])
            y_model = np.array(y_model)

            fig, ax = plt.subplots()
            ax.plot(x_test,y_answer,label='Answer')
            ax.plot(x_test,y_model,label='Model')
            ax.legend()
            plt.xlabel('X'); plt.ylabel('Y')
            plt.grid(axis='both',color='0.95')

        elif input.N_order_x == 2 and input.N_order_y == 2:
            x_test = np.zeros((input.N_order_x,input.N_x_test))
            x_test[0,:] = np.linspace(input.x_test_min[0],input.x_test_max[0],input.N_x_test)
            x_test[1,:] = np.linspace(input.x_test_min[1],input.x_test_max[1],input.N_x_test)
            y_answer = testFunc(x_test,input.N_order_x,input.N_order_y)
            
            y_model = np.zeros((input.N_order_x,input.N_x_test))
            for n_x_test in range(input.N_x_test):
                y_model[:,n_x_test] = dn.in2out([x_test[0][n_x_test], x_test[1][n_x_test]])

            fig, ax = plt.subplots()
            ax.plot(x_test[0],y_answer[0],label='Answer 0')
            ax.plot(x_test[1],y_answer[1],label='Answer 1')
            ax.plot(x_test[0],y_model[0],label='Model 0')
            ax.plot(x_test[1],y_model[1],label='Model 1')
            ax.legend()
            plt.xlabel('X'); plt.ylabel('Y')
            plt.grid(axis='both',color='0.95')
            
        elif input.N_order_x == 2 and input.N_order_y == 1:
            # Mesh grid
            N_1D = int(np.sqrt(input.N_x_test))
            x1D = np.linspace(input.x_test_min[0],input.x_test_max[0],N_1D)
            y1D = np.linspace(input.x_test_min[1],input.x_test_max[1],N_1D)

            x2D, y2D = np.meshgrid(x1D, y1D, indexing='xy')
            x2D_vect = np.reshape(x2D,(-1,))
            y2D_vect = np.reshape(y2D,(-1,))
            N_2D_vect = len(x2D_vect)
            
            # Answer from test function
            x_test = np.zeros((input.N_order_x,N_2D_vect))
            x_test[0,:] = x2D_vect
            x_test[1,:] = y2D_vect
            z_answer = testFunc(x_test,input.N_order_x,input.N_order_y)
            z_answer2D = np.reshape(z_answer,(N_1D,N_1D))
            
            # Network output
            z_model = np.zeros((input.N_order_y,N_2D_vect))
            for n_2D_vect in range(N_2D_vect):
                z_model[:,n_2D_vect] = dn.in2out([x_test[0][n_2D_vect], x_test[1][n_2D_vect]])
            z_model2D = np.reshape(z_model,(N_1D,N_1D))

            # 3D plot
            fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
            ax1.plot_wireframe(x2D, y2D, z_answer2D, color=(0.9,0.3,0.3),label='Answer')
            ax1.scatter(x2D, y2D, z_model2D, color=(0.3,0.3,0.9), marker='o', label='Model')
            plt.legend()
            plt.xlabel('X'); plt.ylabel('Y'); plt.title("Answer vs Model")
            plt.grid(axis='both',color='0.95')
            
            # Error 2D plot
            fig2, ax2 = plt.subplots()
            c = ax2.pcolor(x2D,y2D,z_answer2D-z_model2D, cmap='RdBu', vmin=-1, vmax=+1)
            fig2.colorbar(c, ax=ax2)
            ax2.set_title("Error")
            plt.xlabel('X'); plt.ylabel('Y')
            plt.grid(axis='both',color='0.95')

        else:
            print("No plot type available")
        plt.show()

if __name__ == '__main__':
    ## Profiling main
    t_start = time.perf_counter()
    profileName = 'temp.prof'
    main(profileName)
    
    ## Profiling visualization
    snakeVizCmd = "python -m snakeviz " + profileName
    removeCmd_dos = "del " + profileName
    removeCmd_linux = "rm " + profileName
    p = subprocess.Popen(snakeVizCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    userInput = input("Enter to complete")
    p = subprocess.Popen(removeCmd_dos, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    p = subprocess.Popen(removeCmd_linux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    ## End
    t_end = time.perf_counter()
    print("Elapsed time: {:.3f} seconds".format(t_end-t_start))
