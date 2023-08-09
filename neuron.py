import numpy as np

class neuron():
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
            if co_sum < 0:
                self.val_out =  0
            else:
                self.val_out = co_sum

        elif self.actFuncType == 1:
            # Sigmoidal
            sig = 1/(1+np.exp(-co_sum))
            self.val_out = sig

        else:
            self.val_out = np.NaN
            print("Invalid activation function type: %d" % self.actFuncType)

        return self.val_out

