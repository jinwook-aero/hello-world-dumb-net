import numpy as np

class neuron():
    ## Nueron
    # Each nueron has two parts
    # - Receptor that sums up the output signal from previous layers
    # - Activation function
    # 
    # Inputs:
    # - N_leg = number of upstream nuerons
    # - initWeight = initial weight // np.NaN set random weights
    # - actFuncType = activation function
    #   // -1: No activation function
    #   //  0: Rectified linear unit
    #   //  1: Sigmoidal
    #   //  2: Swish
    #
    # Functions:
    # - getWeights // get weights of all neurons
    # - setWeights(newWeights)// set weights of all neurons
    #   | newWeights = np.zeros((self.N_leg+1),dtype=float)
    # -in2out(val_ins) // compute output of network
    #   | val_ins = np.zeros((self.N_leg),dtype=float)
    # -out2in(dNdx_down) // back-propagate sensitivities
    #
    # Last update: August 20, 2023
    # Author: Jinwook Lee
    #
    def __init__(self, N_leg, initWeight=np.NaN, actFuncType=2):
        # Initialization
        self.N_leg   = N_leg
        self.N_weight = N_leg + 1
        self.val_ins = np.zeros(N_leg)
        self.val_sum = 0 # Weighted sum of input
        self.val_out = 0
        self.actFuncType = actFuncType

        # Sensitivities
        self.dNdx_down = 0 # sensitivity of network output to neuron output
        self.dNdx_leg = np.zeros(N_leg) # sensitivity of network output to each leg input
        self.dNdweight = np.zeros(self.N_weight) # sensitivity of network output to each weight

        # Weights
        self.weights = np.zeros((self.N_weight),dtype=float)
        if np.isnan(initWeight):
            self.weights = 2*np.random.rand(self.N_weight)-1 # rand [-1, 1]
        else:
            self.weights = np.ones(self.N_weight)*initWeight
    
    def getWeights(self):
        return self.weights
    
    def setWeights(self,newWeights):
        self.weights = newWeights
    
    def _computeActFunct(self,co_sum):
        if self.actFuncType == -1:
            # No activation function
            co_out = co_sum

        elif self.actFuncType == 0:
            # ReLU
            co_out = max(co_sum,0)

        elif self.actFuncType == 1:
            # Sigmoidal
            co_out = 1/(1+np.exp(-co_sum))

        elif self.actFuncType == 2:
            # Swish
            sig = 1/(1+np.exp(-co_sum))
            co_out = co_sum * sig

        else:
            co_out = np.NaN
            print("Invalid activation function type: %d" % self.actFuncType)
        
        return co_out
    
    def _computeActGradient(self,co_sum):
        if self.actFuncType == -1:
            # No activation function
            co_out = np.ones(np.size(co_sum))

        elif self.actFuncType == 0:
            # ReLU
            co_out = np.ones(np.size(co_sum))
            i_sel = co_sum<0
            co_out[i_sel] = 0

        elif self.actFuncType == 1:
            # Sigmoidal
            co_exp = np.exp(+co_sum)
            co_sig = co_exp/(1+co_exp)
            co_out = co_sig/(1+co_exp)

        elif self.actFuncType == 2:
            # Swish
            co_exp = np.exp(+co_sum)
            co_sig = co_exp/(1+co_exp)
            co_out = co_sig*(1+co_sum+co_exp)/(1+co_exp)

        else:
            co_out = np.NaN
            print("Invalid activation function type: %d" % self.actFuncType)
        
        return co_out
    
    def in2out(self,val_ins):
        self.val_ins = val_ins
        assert len(self.val_ins) == self.N_leg # Input count must match leg count
        self.val_sum = np.dot(self.weights[:-1],self.val_ins) + self.weights[-1]
        self.val_out = self._computeActFunct(self.val_sum)

        return self.val_out

    def out2in(self,dNdx_down):
        ## Back-propagation of network sensitivity
        self.dNdx_down = dNdx_down
        dydx_neuron = self._computeActGradient(self.val_sum)

        # Sensitivity of network output to upstream legs
        dNdx_mid = dNdx_down*dydx_neuron
        self.dNdx_up = dNdx_mid*self.weights[:-1]

        # Sensitivity of network output to weights
        self.dNdweight[:-1] = dNdx_mid*self.val_ins
        self.dNdweight[-1] = dNdx_mid



