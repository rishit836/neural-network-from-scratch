from .grad import Value,Tensor
import random
import numpy as np



class Neuron_value:
    "deprecated uses basic python operations which is slow."
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w+[self.b]
    
class Layer_value:
    "deprecated uses basic python operations which is slow."
    def __init__(self,nin,nout):
        self.neurons = [Neuron_value(nin) for _ in range(nout)]
    
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) ==1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP_value:
    "deprecated uses basic python operations which is slow."
    def __init__(self,nin,nouts): 
        '''
        nin: number of inputs to the neural network
        nouts: list containing of sizes of hidden layer in order including output layer size and excluding input layersize eg:[2,2,1] for a
               a neural network containing 2 hidden layer with 2 neurons each and one output
        '''
        sz = [nin] + nouts
        self.layers = [Layer_value(sz[i],sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
class Layer:
    def __init__(self, nin, nout):
        # LeCunn Initialization in the Layer Class
        self.W = Tensor(np.random.randn(nin, nout) * np.sqrt(1 / nin))
        self.b = Tensor(np.zeros((1, nout)))

    def __call__(self, x):
        # y = wx + b
        return x @ self.W + self.b
    
    def parameters(self):
        return [self.W,self.b]
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):

        for layer in self.layers[:-1]:
            x = layer(x).tanh()

        x = self.layers[-1](x)

        return x

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params
    
class ClassifcationNN:
    def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        
        for idx,layer in enumerate(self.layers):
            # last layer has softmax
            if idx+1 == len(self.layers):
                x = layer(x).softmax()
            else:
                x = layer(x).tanh()
        
        return x

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

if __name__=='__main__':
    mlp = MLP(3, [4, 4, 1])

    for p in mlp.parameters():
        print(p.data.shape)