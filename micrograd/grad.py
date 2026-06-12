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
        self.data = np.array(data)
        self.grad = np.zeros_like(data)
        self._op = _op
        self._prev = set(_children)
        
        
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
        
        if not isinstance(other, Tensor):
            if isinstance(other, (float,int)):
                otherlike = np.ones_like(self.data)
                other = otherlike * other #converting the data into list so operation can be done
                other = Tensor(data=other)
            else:
                raise ValueError("please make sure the addition operation is between a tensor and a scalar or bw tensor and tensor")
            
            
        out = Tensor(self.data+other.data, _op="+",_children=(self,other))
        
        return out
    
    def __mul__(self,other):
        # element wise operation
        if not isinstance(other,Tensor):
            if isinstance(other, (float,int)):
                otherlike = np.ones_like(self.data)
                other = otherlike * other
                other = Tensor(data = other)
            else:
                raise ValueError("please make sure the multiplication operation is between a tensor and a scalar or bw tensor and tensor")
        out =Tensor(self.data*other.data, _op="*", _children=(self,other))
        return out

if __name__ == "__main__":
    t = Tensor([1,2,3,4])
    t1 = Tensor([1,2,3,4])
    print(t*t1)