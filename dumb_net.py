import numpy as np
from neuron import neuron

class DumbNet():
    def __init__(self,n_layer_dist,initWeight=np.NaN,actFuncType=0):
        # Last neuron check
        assert n_layer_dist[-1] == 1 # Last layer is neuron without activation

        # Initialization
        self.N_width = n_layer_dist # [n_layer][n_width] // width of each layer
        self.N_layer = len(self.N_width)

        self.actFuncType = actFuncType
        self._createNet(initWeight,actFuncType) # create self.neurons

        self.N_neuron = sum(self.N_width) # Total count of nuerons
        self.N_weight = 0 # Total count of weights
        for n_layer in range(self.N_layer):
            for n_width in range(self.N_width[n_layer]):
                cur_neuron = self.neurons[n_layer][n_width]
                self.N_weight += len(cur_neuron.getWeights())

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