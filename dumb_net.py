import numpy as np
from neuron import neuron

class DumbNet():
    ## Dumb network that takes single input to set single output
    # Network established based on user defined list
    # Provides simple interface to get and set weights for all neurons
    #
    # Input: 
    # - n_layer_dist = np.array([a,b,c,...,d,1])
    #   | Must end with 1 (last layer to sum results)
    #   | Each layer should not be zero
    #
    # Functions:
    # - in2out(x_in) // Computes output of network
    # - getWeights
    # - train(input,testFunc,errorFunc)
    #   | input.eps # perturbation to compute gradient
    #   | input.tol # Acceptable range within which training is skipped
    #   | input.conv # Convergence criterion
    #   | input.N_iter # Iteration count
    #   | input.x_train_min # Train x range, min
    #   | input.x_train_max # Train x range, max
    #   | input.N_x_train # Train x count
    #   | input.weightTrainFrac # Fraction of weight to train per each iteration
    #   | input.weightLearnRate # Weight learning rate
    #   | input.xFocusFrac      # Fraction of test x to focus on per each iteration
    def __init__(self,n_layer_dist,initWeight=np.NaN,actFuncType=0):
        # Layer definition check
        assert n_layer_dist[-1] == 1 # Last layer is neuron without activation
        assert np.all(n_layer_dist>0) # All layer size should be positive

        # Initialization
        self.N_width = n_layer_dist # self.N_width[n_layer] width of n_layer
        self.N_layer = len(self.N_width)

        self.actFuncType = actFuncType
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
            if n_layer == 0:
                cur_in = np.array([x_in])
            else:
                cur_in = np.array(cur_out)

            # Passing neurons
            cur_out = []
            for neuron in self.neurons[n_layer]:
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
    
    def setWeights(self,newWeights):
        # Neuron weights
        iEnd = -1
        for n_layer in range(self.N_layer):            
            for n_width in range(self.N_width[n_layer]):
                cur_neuron = self.neurons[n_layer][n_width]
                i0 = iEnd+1
                iEnd += (cur_neuron.N_leg + 1) # count of weights on this neuron
                cur_neuron.setWeights(newWeights[i0:iEnd+1])
    
    def _createNet(self,initWeight,actFuncType):
        self.neurons = np.ndarray((self.N_layer,),dtype=neuron)
        for n_layer in range(self.N_layer):
            self.neurons[n_layer] = np.ndarray((self.N_width[n_layer],),dtype=neuron)
            
            N_leg = 1 if n_layer == 0 else self.N_width[n_layer-1]
            if n_layer == self.N_layer-1: # Last layer is always neuron without activation
                self.neurons[n_layer][0] = neuron(N_leg,initWeight,-1)
            else:
                for n_width in range(self.N_width[n_layer]):
                    self.neurons[n_layer][n_width] = neuron(N_leg,initWeight,actFuncType)

    def train(self,input,testFunc,errorFunc):
        # Inputs
        n_iter_list = range(input.N_iter)
        x_in_list = np.linspace(input.x_train_min,input.x_train_max,input.N_x_train)

        # Iterations
        eMin_list = []
        eMax_list = []
        eAvg_list = []
        for n_iter in n_iter_list:
            # Selected x list with worst error
            error_list = np.empty((0),dtype=float)
            for x_in in x_in_list:
                # Current error
                # - Skip if already within tolerance
                e_init = errorFunc(self,testFunc,x_in)
                error_list = np.append(error_list,e_init)
            i_sort = np.argsort(error_list)
            i_sel = i_sort[-int(input.N_x_train*input.xFocusFrac):]
            x_sel = x_in_list[i_sel]

            # Update based on worst x cases
            for x_in in x_sel:
                # Current error
                # - Skip if already within tolerance
                e_init = errorFunc(self,testFunc,x_in)
                if abs(e_init)<input.tol:
                    continue
                
                # Random selection of weights to train
                n_w_all = np.arange(self.N_weight)
                N_w_sel = int(np.floor(self.N_weight*input.weightTrainFrac))
                n_w_sel = np.random.choice(n_w_all, N_w_sel, replace=False)  

                # Compute Jacobian
                oldWeights = self.getWeights()
                co_J = np.empty((0,1),dtype=float)
                for n_w in n_w_sel:
                    newWeights = np.copy(oldWeights)
                    newWeights[n_w] += input.eps
                    self.setWeights(newWeights)
                        
                    e_updt = errorFunc(self,testFunc,x_in)
                    dedx = (e_updt-e_init)/input.eps
                    co_J = np.append(co_J,dedx)
                co_J2D = co_J.reshape(1,len(co_J))
                co_err2D = e_init.reshape(1,1)

                co_b2D = -(co_J2D.transpose()) @ co_err2D
                co_a2D = (co_J2D.transpose()) @ co_J2D
                
                # Update globally
                dx2D = np.linalg.lstsq(co_a2D, co_b2D, rcond=None)[0]
                dx1D = dx2D.reshape(-1,)
                dWeights = np.zeros(self.N_weight)
                dWeights[n_w_sel] = dx1D*input.weightLearnRate
                solWeights = oldWeights+dWeights
                self.setWeights(solWeights)

            ## Iteration status
            # Display
            eMin_list.append(min(error_list))
            eMax_list.append(max(error_list))   
            eAvg_list.append(sum(error_list)/len(error_list))        
            print("Iteration {}, minE={:.2e}, maxE={:.2e}, avgE={:.2e}".format(
                n_iter,min(error_list),max(error_list),sum(error_list)/len(error_list)))
            
            ## Stagnation check
            # Breakout if not converging
            # for more than 10% of total iteration
            if n_iter == 0:
                eAvg0 = eAvg_list[0]
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
        if eAvg_list[-1]<input.conv:
            self.isTrained = True
        else:
            self.isTrained = False

        # Return convergence history
        return [eMin_list, eMax_list, eAvg_list]
