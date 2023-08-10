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
    #
    # Functions:
    # - getWeights // get weights of all neurons
    # - setWeights(newWeights)// set weights of all neurons
    #   | newWeights = np.zeros((self.N_leg+1),dtype=float)
    # -in2out(val_ins) // compute output of network
    #   | val_ins = np.zeros((self.N_leg),dtype=float)
    def __init__(self, N_leg, initWeight=np.NaN, actFuncType=0):
        self.N_leg   = N_leg
        self.val_ins = np.zeros(N_leg)
        self.val_out = 0
        self.actFuncType = actFuncType

        # Weights
        self.weights = np.zeros((self.N_leg+1),dtype=float)
        if np.isnan(initWeight):
            self.weights = 2*np.random.rand(N_leg+1)-1 # rand [-1, 1]
        else:
            self.weights = np.ones(N_leg+1)*initWeight
    
    def getWeights(self):
        return self.weights
    
    def setWeights(self,newWeights):
        self.weights = newWeights
    
    def in2out(self,val_ins):
        self.val_ins = val_ins
        assert len(self.val_ins) == self.N_leg # Input count must match leg count
        co_sum = np.dot(self.weights[:-1],self.val_ins) + self.weights[-1]
        if self.actFuncType == -1:
            # No activation function
            self.val_out = co_sum

        elif self.actFuncType == 0:
            # ReLU
            self.val_out = max(co_sum,0)

        elif self.actFuncType == 1:
            # Sigmoidal
            self.val_out = 1/(1+np.exp(-co_sum))

        else:
            self.val_out = np.NaN
            print("Invalid activation function type: %d" % self.actFuncType)

        return self.val_out

