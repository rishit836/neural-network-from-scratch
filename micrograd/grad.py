import math

import numpy as np
class Value:
    def __init__(self,data,_op='',_children=(),label=''):
        self.data = data
        self._op = _op
        self._prev = set(_children)
        self.label = label
        self.grad = 0
        self._backward = lambda:None
        
    def __repr__(self)    :
        return f"Value(data={self.data})"
    
    # other + self
    def __radd__(self, other):
        return self.__add__(other)
    
    # self + other
    def __add__(self,other):
        if not isinstance(other,Value):
            other = Value(other)
        out = Value(self.data+other.data, '+',(self,other))
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self,other):
        if not isinstance(other,Value):
            other = Value(other)
        out = Value(self.data*other.data, '*',(self,other))
        
        def _backward():
            self.grad += other.data* out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other): #other * self
        return self.__mul__(other)
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, _children=(self, ), _op='tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), _children=(self, ), _op='exp')
        
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
    
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()



class Tensor:
    def __init__(self,data:np.ndarray,_op='',_children=()):
        self.data = np.array(data,dtype=np.float64)
        self.grad = np.zeros_like(data,dtype=np.float64)
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda:None
        
        
    def __repr__(self):
        return f"Tensor(data={self.data})"
    
    
    # scalar + tensor
    def __radd__(self, other):
        return self + other
    
    # tensor + scalar
    def __add__(self, other):
    
        """
        the addition method for adding scalars and tensors together
        """
        # if addition is between a floating or integer.
        if not isinstance(other, Tensor) and isinstance(other,(float,int)):
            other = Tensor(data=other)
            
        
        # if not isinstance(other, Tensor):
        #     if isinstance(other, (float,int)):
        #         otherlike = np.ones_like(self.data)
        #         other = otherlike * other #converting the data into list so operation can be done
        #         other = Tensor(data=other)
        #     else:
        #         raise ValueError("please make sure the addition operation is between a tensor and a scalar or bw tensor and tensor")
            
            
        out = Tensor(self.data+other.data, _op="+",_children=(self,other))
        
        # unbroadcasting the broadcast done by numpy in forward pass 
        def unbroadcast(out_grad,data_shape):
            # collapsing all the rows into one by adding the corresponding columns into one.
            if len(out_grad.shape) > len(data_shape): #checking if the number of rows are more than the data.
                out_grad = out_grad.sum(axis=0)
                
            # collapsing columns into incase the shape has dimension of 1
            for i,dim in enumerate(data_shape):
                if dim == 1:
                    out_grad = out_grad.sum(axis=i,keepdims=True)
                    
            return out_grad
        
        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
            
        out._backward = _backward
        
        
        return out
    
    def __mul__(self,other):
        
        # if multiplication is between a floating or integer.
        if not isinstance(other, Tensor) and isinstance(other,(float,int)):
            other = Tensor(data=other)
        
        # element wise operation
        # if not isinstance(other,Tensor):
        #     if isinstance(other, (float,int)):
        #         otherlike = np.ones_like(self.data)
        #         other = otherlike * other
        #         other = Tensor(data = other)
        #     else:
        #         raise ValueError("please make sure the multiplication operation is between a tensor and a scalar or bw tensor and tensor")
        
        out =Tensor(self.data*other.data, _op="*", _children=(self,other))
        
        # unbroadcasting the broadcast done by numpy in forward pass 
        def unbroadcast(out_grad,data_shape):
            # collapsing all the rows into one by adding the corresponding columns into one.
            if len(out_grad.shape) > len(data_shape): #checking if the number of rows are more than the data.
                out_grad = out_grad.sum(axis=0)
                
            # collapsing columns into incase the shape has dimension of 1
            for i,dim in enumerate(data_shape):
                if dim == 1:
                    out_grad = out_grad.sum(axis=i,keepdims=True)
                    
            return out_grad
        
        
        def _backward():
            self.grad += unbroadcast(other.data * out.grad,self.data.shape)
            other.grad += unbroadcast(self.data * out.grad,other.data.shape)
        
        out._backward = _backward
        
        return out
    
if __name__ == "__main__":
    x = Tensor([[10,20,30],[30,10,50]])
    b = Tensor([[5,5,5]])
    y = x+b
    y.grad = np.ones_like(y.data,dtype=np.float64)
    y._backward()
    x._backward()
    b._backward()
    print(b.grad)