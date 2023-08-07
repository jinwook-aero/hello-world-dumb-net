import numpy as np

class neuron():
    def __init__(self, N_leg, initWeight=1, actFuncType=0):
        self.N_leg   = N_leg
        self.val_ins = np.zeros(N_leg)
        self.val_sum = 0
        self.val_out = 0
        self.actFuncType = actFuncType
        if initWeight < 0:
            self.weights = 2*np.random.rand(N_leg)-1
        else:
            self.weights = np.ones(N_leg)*initWeight

    def getOut(self):
        return self.val_out
    
    def getWeights(self):
        return self.weights
    
    def setWeights(self,newWeights):
        self.weights = newWeights
    
    def in2out(self,val_ins):
        self.val_ins = val_ins
        assert len(self.val_ins) == self.N_leg
        self.val_sum = np.sum(self.val_ins*self.weights)
        if self.actFuncType == 0:
            # ReLU
            if self.val_sum < 0:
                self.val_out =  0
            else:
                self.val_out = self.val_sum

        elif self.actFuncType == 1:
            # Sigmoidal
            sig = 1/(1+np.exp(-self.val_sum))
            self.val_out = sig

        else:
            self.val_out = np.NaN
            print("Invalid activation function type: %d" % self.actFuncType)

        return self.val_out

