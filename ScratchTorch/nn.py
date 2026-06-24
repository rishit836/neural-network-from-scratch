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
    
class Dense:
    def __init__(self, nin, nout):
        # LeCunn Initialization in the Layer Class
        self.W = Tensor(np.random.randn(nin, nout) * np.sqrt(1 / nin))
        self.b = Tensor(np.zeros((1, nout)))
        self.activation = 'tanh'

    def __call__(self, x):
        # y = wx + b
        y = x @ self.W + self.b
        if self.activation == 'softmax':
            return y.softmax()
        else:
            return y.tanh()
    
    def parameters(self):
        return [self.W,self.b]
    
    def set_activation_function(self,activation):
        self.activation = activation
        return self
    
    

            
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Dense(sz[i], sz[i+1])
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
        self.layers = [Dense(sz[i],sz[i+1]) for i in range(len(nouts))]
        
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
    
# Convulution Neural Network.
# convultuion layer
class Conv2d:
    def __init__(self,in_channels,out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.layer = Tensor(np.random.uniform(-1,1,(self.out_channels,self.in_channels,self.kernel_size,self.kernel_size)))
        self.b = Tensor(np.zeros((1,out_channels)))
    
    def __call__(self,x:Tensor):
        feature_map = []
        for idx,filter in enumerate(self.layer):
            _out = x.conv2d(filter)
            feature_map.append(_out)
        output_tensor = Tensor.stack(feature_map)
        return output_tensor



if __name__=='__main__':
    mlp = MLP(3, [4, 4, 1])

    for p in mlp.parameters():
        print(p.data.shape)