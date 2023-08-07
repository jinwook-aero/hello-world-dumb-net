import numpy as np
from neuron import neuron

class DumbNet():
    def __init__(self,n_layer_dist,initWeight=1,actFuncType=0):
        self.N_width = n_layer_dist
        self.N_layer = len(self.N_width)

        self.actFuncType = actFuncType
        self.neurons = self._createNet(initWeight,actFuncType)

        self.N_neuron = sum(self.N_width)
        self.N_weight = 0
        for n_layer in range(self.N_layer):
            for n_width in range(self.N_width[n_layer]):
                N_leg = 1 if n_layer == 0 else self.N_width[n_layer-1]
                self.N_weight += N_leg

    def in2out(self,x_in):
        for n_layer in range(self.N_layer):
            if n_layer == 0:
                cur_in = np.array([x_in])
            else:
                cur_in = cur_out

            cur_out = []
            for neuron in self.neurons[n_layer]:
                cur_out.append(neuron.in2out(cur_in))
            cur_out = np.array(cur_out)
        return cur_out[0]

    def getWeights(self):
        weights = []
        for n_layer in range(self.N_layer):            
            for n_width in range(self.N_width[n_layer]):
                cur_neuron = self.neurons[n_layer][n_width]
                weights.append(list(cur_neuron.getWeights()))
        return np.hstack(weights)
    
    def setWeights(self,newWeights):
        iEnd = -1
        for n_layer in range(self.N_layer):            
            for n_width in range(self.N_width[n_layer]):
                cur_neuron = self.neurons[n_layer][n_width]
                i0 = iEnd+1
                iEnd += len(cur_neuron.getWeights())
                newWeightsSel = newWeights[i0:iEnd+1]
                cur_neuron.setWeights(newWeightsSel)
    
    def _createNet(self,initWeight,actFuncType):
        neurons = np.ndarray((self.N_layer,),dtype=neuron)
        for n_layer in range(self.N_layer):
            neurons[n_layer] = np.ndarray((self.N_width[n_layer],),dtype=neuron)
            
            N_leg = 1 if n_layer == 0 else self.N_width[n_layer-1]
            for n_width in range(self.N_width[n_layer]):
                neurons[n_layer][n_width] = neuron(N_leg,initWeight,actFuncType)
        return neurons