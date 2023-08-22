import numpy as np
import time
from neuron import neuron
import numba as nb

class DumbNet():
    ## Dumb network that takes vector input to set vector output
    # Network established based on user defined list
    # Provides simple interface to get and set weights for all neurons
    #
    # Input: 
    # - n_layer_dist = np.array([N_in,a,b,c,...,d,N_out])
    #   | Each layer should not be zero
    #
    # Functions:
    # - in2out(x_in) // Computes output of network
    # - getWeights
    # - get_dydx_weights // Sensitivity of network output to weights
    # - setWeights
    # - train(input,testFunc,errorFunc)
    #   | input.eps # perturbation to compute gradient
    #   | input.tol # Acceptable range within which training is skipped
    #   | input.conv # Convergence criterion
    #   | input.N_iter # Iteration count
    #   | input.x_train_min # Train x range, min
    #   | input.x_train_max # Train x range, max
    #   | input.N_x_train # Train x count
    #   | input.N_x_test # Test x count
    #   | input.weightTrainFrac # Fraction of weight to train per each iteration
    #   | input.weightLearnRate # Weight learning rate
    #
    # Last update: August 21, 2023
    # Author: Jinwook Lee
    #
    def __init__(self,n_layer_dist,initWeight=np.NaN,actFuncType=0,eps=1E-10):
        # Layer size check
        assert np.all(n_layer_dist>0) # All layer size should be positive

        # Initialization
        self.N_width = n_layer_dist # self.N_width[n_layer] width of n_layer
        self.N_first = n_layer_dist[0]
        self.N_last = n_layer_dist[-1]
        self.N_layer = len(self.N_width)

        self.actFuncType = actFuncType
        self.eps = eps # Perturbation to compute sensitivity
        self._createNet(initWeight,actFuncType) # create self.neurons

        self.N_neuron = sum(self.N_width) # Total count of nuerons
        self.N_weight = 0 # Total count of weights
        for n_layer in range(self.N_layer):
            for n_width in range(self.N_width[n_layer]):
                cur_neuron = self.neurons[n_layer][n_width]
                self.N_weight += len(cur_neuron.getWeights())
        print("Built network with {} neurons and {} weights".format(self.N_neuron,self.N_weight))
        
        # Train status
        self.isTrained = False

    def in2out(self,x_in):
        for n_layer in range(self.N_layer):
            # Input
            if n_layer != 0:
                cur_in = cur_out
            #else:
            #    to be defined based on x_in[n_width]

            # Compute signal through all neurons on n_layer'th layer
            cur_out = []
            for n_width in range(self.N_width[n_layer]):
                if n_layer == 0:
                    cur_in = np.array([x_in[n_width]])
                neuron = self.neurons[n_layer][n_width]
                cur_out.append(neuron.in2out(cur_in))
            cur_out = np.array(cur_out)
        return cur_out

    def getWeights(self):
        # Neuron weights
        weights = np.empty((0),dtype=float)
        for n_layer in range(self.N_layer):            
            for n_width in range(self.N_width[n_layer]):
                cur_neuron = self.neurons[n_layer][n_width]
                weights = np.append(weights,cur_neuron.getWeights(),axis=0)
        return weights
    
    def get_dNdweights(self):
        # Network sensitivity to weights
        dNdweights = np.empty((0,self.N_last),dtype=float)
        for n_layer in range(self.N_layer):            
            for n_width in range(self.N_width[n_layer]):
                cur_neuron = self.neurons[n_layer][n_width]
                dNdweights = np.append(dNdweights,cur_neuron.dNdweight,axis=0)
        return dNdweights
    
    def setWeights(self,newWeights):
        # Neuron weights
        iEnd = -1
        for n_layer in range(self.N_layer):            
            for n_width in range(self.N_width[n_layer]):
                cur_neuron = self.neurons[n_layer][n_width]
                i0 = iEnd+1
                iEnd += (cur_neuron.N_in + 1) # count of weights on this neuron
                cur_neuron.setWeights(newWeights[i0:iEnd+1])
    
    def _createNet(self,initWeight,actFuncType):
        self.neurons = np.ndarray((self.N_layer,),dtype=neuron)
        for n_layer in range(self.N_layer):
            self.neurons[n_layer] = np.ndarray((self.N_width[n_layer],),dtype=neuron)
            
            N_in = 1 if n_layer == 0 else self.N_width[n_layer-1]
            if n_layer == self.N_layer-1: # Last layer is always neuron without activation
                curActFuncType = -1
            else:
                curActFuncType = actFuncType
            for n_width in range(self.N_width[n_layer]):
                self.neurons[n_layer][n_width] = neuron(N_in,self.N_last,initWeight,curActFuncType)
 
    def train(self,input,testFunc,errorFunc):
        # Inputs
        n_iter_list = range(input.N_iter)
        x_in_list = []
        for n_order_x in range(input.N_order_x):
            x_in_list.append(np.linspace(\
                                input.x_train_min[n_order_x],\
                                input.x_train_max[n_order_x],\
                                input.N_x_train))

        # Iterations
        eMin_list = np.full([input.N_iter], np.nan)
        eMax_list = np.full([input.N_iter], np.nan)
        eAvg_list = np.full([input.N_iter], np.nan)
        dt_forward = 0
        dt_backward = 0
        dt_leastsquare = 0
        for n_iter in n_iter_list:
            # Forward propagate to compute error based on randomly selected x
            # - Skip if already within tolerance
            cur_time1 = time.perf_counter()
            x_in = np.zeros(input.N_order_x)
            for n_order_x in range(input.N_order_x):
                i_sel = np.random.randint(0, len(x_in_list[n_order_x]))
                x_in[n_order_x] = x_in_list[n_order_x][i_sel]
            y_est = self.in2out(x_in)
            y_ans = testFunc(x_in,self.N_first,self.N_last)
            e_init = errorFunc(y_ans,y_est)
            
            # Sensitivity of error to network output
            dedout = np.zeros((input.N_order_y,1))
            for n_order_y in range(input.N_order_y):
                y_pert = y_est[:]
                y_pert[n_order_y] += self.eps
                e_pert = errorFunc(y_ans,y_pert)
                dedout[n_order_y] = (e_pert-e_init)/(self.eps)
            cur_time2 = time.perf_counter()
            dt_forward += cur_time2 - cur_time1

            # Back propagate to compute error sensitivity to weights
            cur_time1 = time.perf_counter()
            for n_layer in reversed(range(self.N_layer)):
                for n_width in range(self.N_width[n_layer]):
                    if n_layer == self.N_layer-1: # Last layer
                        dNdx_out = np.zeros(input.N_order_y)
                        dNdx_out[n_width] = 1
                    else:
                        dNdx_out = 0
                        for n_width2 in range(self.N_width[n_layer+1]):
                            dNdx_out += self.neurons[n_layer+1][n_width2].dNdx_in[n_width]
                    self.neurons[n_layer][n_width].out2in(dNdx_out)
            dedweights = self.get_dNdweights() @ dedout
            cur_time2 = time.perf_counter()
            dt_backward += cur_time2- cur_time1

            # Random selection of weights to train
            cur_time1 = time.perf_counter()
            n_w_all = np.arange(self.N_weight)
            N_w_sel = int(np.floor(self.N_weight*input.weightTrainFrac))
            n_w_sel = np.random.choice(n_w_all, N_w_sel, replace=False)  

            # Compute least-square solution
            co_J = dedweights[n_w_sel]
            co_J2D = co_J.reshape(1,len(co_J))
            co_err2D = e_init.reshape(1,1)
            co_b2D = -(co_J2D.transpose()) @ co_err2D
            co_a2D = (co_J2D.transpose()) @ co_J2D
            dx2D = np.linalg.lstsq(co_a2D, co_b2D, rcond=None)[0]
            dx1D = dx2D.reshape(-1,)
            
            # Update weights
            dWeights = np.zeros(self.N_weight)
            dWeights[n_w_sel] = dx1D*input.weightLearnRate

            oldWeights = self.getWeights()
            solWeights = oldWeights+dWeights
            self.setWeights(solWeights)
            cur_time2 = time.perf_counter()
            dt_leastsquare += cur_time2- cur_time1

            ## Iteration status
            if np.remainder(n_iter,200)==0:
                error_list = np.empty((0),dtype=float)
                
                for n_test in range(input.N_x_test):
                    x_in = np.zeros(input.N_order_x)
                    for n_order_x in range(input.N_order_x):
                        i_sel = np.random.randint(0, len(x_in_list[n_order_x]))
                        x_in[n_order_x] = x_in_list[n_order_x][i_sel]

                    y_est = self.in2out(x_in)
                    y_ans = testFunc(x_in,self.N_first,self.N_last)
                    e_init = errorFunc(y_ans,y_est)
                    error_list = np.append(error_list,e_init)

                # Display
                eMin_list[n_iter] = min(error_list)
                eMax_list[n_iter] = max(error_list)
                eAvg_list[n_iter] = sum(error_list)/len(error_list) 
                print("Iteration {}, minE={:.2e}, maxE={:.2e}, avgE={:.2e}, stdE={:.2e}"
                      .format(n_iter,min(error_list),max(error_list),
                              sum(error_list)/len(error_list),np.std(error_list)))
                
                if n_iter !=0 and np.remainder(n_iter,200*5)==0:
                    dt_total = dt_forward+dt_backward+dt_leastsquare
                    tF_forw = dt_forward/dt_total*100
                    tF_back = dt_backward/dt_total*100
                    tF_lq = dt_leastsquare/dt_total*100
                    print("\tTime Fraction: ForwP={:.1f}%, BackP={:.1f}%, LeastSQ={:.1f}%"
                          .format(tF_forw,tF_back,tF_lq))
                
                ## Stagnation check
                # Breakout if not converging
                # for more than 10% of total iteration
                if n_iter == 0:
                    eAvg0 = eAvg_list[n_iter]
                    n_stagnant = 0
                else:
                    if eAvg_list[n_iter]>=eAvg_list[n_iter-1]: # Not converging
                        n_stagnant += 1
                if n_stagnant > input.N_iter*0.5:
                    break

            ## Convergence check
            if eAvg_list[n_iter] < input.conv:
                break
        
        # Training quality
        if eAvg_list[n_iter]<input.conv:
            self.isTrained = True
        else:
            self.isTrained = False

        # Return convergence history
        return [eMin_list, eMax_list, eAvg_list]
