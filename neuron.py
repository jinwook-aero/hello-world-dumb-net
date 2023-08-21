import numpy as np

class neuron():
    ## Nueron
    # Each nueron has two parts
    # - Receptor that sums up the output signal from previous layers
    # - Activation function
    # 
    # Inputs:
    # - N_in = number of upstream nuerons
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
    #   | newWeights = np.zeros((self.N_in+1),dtype=float)
    # -in2out(val_ins) // compute output of network
    #   | val_ins = np.zeros((self.N_in),dtype=float)
    # -out2in(dNdx_down) // back-propagate sensitivities
    #
    # Last update: August 21, 2023
    # Author: Jinwook Lee
    #
    def __init__(self, N_in, N_last, initWeight=np.NaN, actFuncType=2):
        # Initialization
        self.N_in  = N_in # Count of upstream inputs to neuron
        self.N_last = N_last # Width of last layer of network (i.e. dimension of output vector)
        self.N_weight = N_in + 1
        self.val_ins = np.zeros(N_in)
        self.val_sum = 0 # Weighted sum of input
        self.val_out = 0
        self.actFuncType = actFuncType

        # Sensitivities of network output to:
        # - neuron output
        # - Each upstream input
        # - Each weight
        # // Network last layer has self.N_last neurons
        self.dNdx_out = np.zeros((1,self.N_last))
        self.dNdx_in = np.zeros((self.N_in,self.N_last))
        self.dNdweight = np.zeros((self.N_weight,self.N_last))

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
        assert len(self.val_ins) == self.N_in # Input count must be consistent
        self.val_sum = np.dot(self.weights[:-1],self.val_ins) + self.weights[-1]
        self.val_out = self._computeActFunct(self.val_sum)

        return self.val_out

    def out2in(self,dNdx_out):
        ## Back-propagation of network sensitivity
        self.dNdx_out = dNdx_out
        dydx_neuron = self._computeActGradient(self.val_sum)

        # Sensitivity of network output to upstream legs
        dNdx_mid = dNdx_out*dydx_neuron # // np.NaN((1,self.N_last))
        self.dNdx_in = dNdx_mid*np.reshape(self.weights[:-1],(self.N_in,1)) # // np.NaN((self.N_in,self.N_last))

        # Sensitivity of network output to weights
        # - dNdweight = np.NaN((self.N_in+1,self.N_last))
        self.dNdweight[:-1,:] = dNdx_mid*np.reshape(self.val_ins,(self.N_in,1)) # // np.NaN((self.N_in,self.N_last))
        self.dNdweight[-1,:] = dNdx_mid 



