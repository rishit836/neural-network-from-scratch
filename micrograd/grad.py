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
        # converting the data into a numpy array
        self.data = np.array(data,dtype=np.float64)
        # creating the gradient array for each element in the array
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        # operation: only valid if the value is not defined and is resultant of some operation
        self._op = _op
        # saving the child into the object creating a graph thus we can backpropagate
        self._prev = set(_children)
        # local backward function which is None by default because value can be constant thus no default derivative
        self._backward = lambda:None
        
        
    def __repr__(self):
        # representation string
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
        # converting the data into a tensor to have a similar data type
        if not isinstance(other,Tensor):
            if isinstance(other,(float,int,list)):
                other = Tensor(data=other)
            else:
                raise ValueError(f"Addition can only be done between integer/float or array. [passed dtype:{type(other)}]")
            
        out = Tensor(self.data+other.data, _op="+",_children=(self,other))
        
        # unbroadcasting the broadcast done by numpy in forward pass 
        def unbroadcast(out_grad, data_shape):
            # if the data has rows then we collapse the rows into one row by adding all the rows in horizontal axis.
            while len(out_grad.shape) > len(data_shape):
                out_grad = out_grad.sum(axis=0)

            # incase the desired data shape has 1 dimension we collapse that dimension thus unbroadcasting the array.
            for i, dim in enumerate(data_shape):
                if dim == 1:
                    out_grad = out_grad.sum(axis=i, keepdims=True)
                    
            return out_grad
        
        def _backward():
            # derivative of addition is 1, we just need to convert the shape of the gradient into desired shape
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
            
        out._backward = _backward
        
        
        return out
    
    # subtraction
    # negation (conversion of the value into negative)
    def __neg__(self):
        return self * -1
    # subtraction is just adding negative value and positive thus simplfying the propagation
    # self - other
    def __sub__(self, other):
        return self + (-other)
    
    # other - self
    def __rsub__(self, other):
        return other + (-self)
    
    # multiplication
    # self * other
    def __mul__(self,other):
        
        # if multiplication is between a floating or integer or list.
        if not isinstance(other,Tensor):
            if isinstance(other,(float,int,list)):
                other = Tensor(data=other)
            else:
                raise ValueError("Multiplication can only be done between integer/float or array. [passed dtype:{type(other)}]")
        
        out =Tensor(self.data*other.data, _op="*", _children=(self,other))
        
        # unbroadcasting the broadcast done by numpy in forward pass 
        def unbroadcast(out_grad, data_shape):
            # if the data has rows then we collapse the rows into one row by adding all the rows in horizontal axis.
            while len(out_grad.shape) > len(data_shape):
                out_grad = out_grad.sum(axis=0)

            # incase the desired data shape has 1 dimension we collapse that dimension thus unbroadcasting the array.
            for i, dim in enumerate(data_shape):
                if dim == 1:
                    out_grad = out_grad.sum(axis=i, keepdims=True)
                    
            return out_grad
        
        
        def _backward():
            # derivative is the other value in the multiplication
            # we just convert the shape back to desired shape of the gradient
            self.grad += unbroadcast(other.data * out.grad,self.data.shape)
            other.grad += unbroadcast(self.data * out.grad,other.data.shape)
        
        out._backward = _backward
        
        return out
    
    
    # power
    def __pow__(self,power):
        # only supporting the power of integer.
        assert isinstance(power, (int)) 
        
        out = Tensor(self.data ** power, _op=f"**{power}",_children=(self,) )
        
        def _backward():
            # d(nx)/dx = nx**(n-1)
            self.grad += (power* (self.data**(power-1))) * out.grad
        out._backward = _backward
        
        return out
    
    # division
    
    # self / other
    def __truediv__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(data=other)
        return self *(other**-1)
    # other / self
    def __rtruediv__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(data=other)
        return other *(self**-1)
    
    
    # exp
    def exp(self):
        out = Tensor(data=np.exp(self.data), _op="exp", _children=(self,))
        def _backward():
            self.grad += out.data*out.grad
            
        out._backward=_backward
        return out
    
    def tanh(self):
        x =  self.data
        t = t = np.tanh(x)
        out = Tensor(data=t, _op= 'tanh',_children=(self,))
        def _backward():
            self.grad += (1 - t**2) * out.grad
            
        out._backward = _backward
        return out
    
    # Relu
    def relu(self):
        # relu(x)=max(0,x)
        out = Tensor(np.maximum(0, self.data),_op="relu",_children=(self,))

        def _backward():
            # derivative or Relu is 1 if value is postive else 0.
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out
    
    # log
    def log(self):
        # log(x) (using default numpy operation because they fast af boi.)
        # defining epsilion to clip the values so we dont run intpo infinite wala error.
        eps = 1e-12
        safe_data = np.clip(self.data, eps, None)
        out = Tensor(np.log(safe_data),_op="log",_children=(self,))

        def _backward():
            # derivative of log(x) is 1/x
            self.grad += (1/safe_data) * out.grad

        out._backward = _backward

        return out
    
    # SOFTMAX
    def softmax(self):
        # shifted the data points towards 0. thus power/exponation is not very large thus it wont explode numerically.
        axis = 1 if self.data.ndim > 1 else 0
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        # finding e^x
        exp = np.exp(shifted)
        # finding probabilities.
        probs = exp/np.sum(exp, axis=axis, keepdims=True)
        

        out = Tensor(probs, _op="softmax",_children=(self,))
        
        
        def _backward():
            
            g = out.grad
            s = out.data
            # using formula used in pytorch libaries to save on time complexity to N for N^2
            self.grad += s * (g - np.sum(g * s, axis=axis, keepdims=True))
            
        out._backward = _backward
        
        return out
    
    def sum(self, axis=None, keepdims=False):
        """
        np.sum() but had to explicitly code this because we need to store the graph relations between each value 
        for calculating gradient and when we directly use np.sum() the shape of the resultant matrix is determined
        by numpy internally but while backward pass we need the original shape thus doing it manually.
        """
        
        out = Tensor(np.sum(self.data,axis=axis,keepdims=keepdims), _op="sum",_children=(self,))

        def _backward():
            grad = out.grad
            # this checks if while summation any axis was removed.
            if axis is not None and not keepdims:
                # if dimensions were removed while summation
                if isinstance(axis, int):
                    # convert the axis into a tuple for numpy
                    axes = (axis,)
                else:
                    axes = axis

                # reinstert the deleted dimension in the axis it was removed
                for ax in sorted(axes):
                    grad = np.expand_dims(
                        grad,
                        axis=ax
                    )
            # accumalting the gradients.
            self.grad += np.broadcast_to(
                grad,
                self.data.shape
            )

        out._backward = _backward

        return out
            
    # implementing the matrix multiplication so we can directly implement the layer class.
    def __matmul__(self, other):
        # forward pass relies on numpy for faster compute.
        out = Tensor(self.data@other.data,_op="@", _children=(self,other))
        def _backward():
            # relying on wikipedia formula for derivative idk about this one. :-)
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            
        out._backward = _backward
        
        return out
    
    
    # backpropagation
    # build a topo sort and then reverses it so we can propagate the tensors.
    def backward(self): 
        topo = []
        visited =set()
        def build(node):
            if node not in visited:
                visited.add(node)
                
                for child in node._prev:
                    build(child)
                    
            
                topo.append(node)
        self.grad = np.ones_like(self.data,dtype=np.float64)
        build(self)
        
        for node in reversed(topo):
            node._backward()
    
    
    # resetting the gradient in training loop made the code untidy thus implemented this a faster one line code.
    def reset_gradients(self):
        topo = []
        visited =set()
        def build(node):
            if node not in visited:
                visited.add(node)
                
                for child in node._prev:
                    build(child)
                    
            
                topo.append(node)

        build(self)
        
        for node in reversed(topo):
            node.grad = np.zeros_like(node.data,dtype=np.float64)


    
if __name__ == "__main__":
    x = Tensor([[10,20,30],[30,10,50]])
    b = Tensor([[5,5,5]])
    y = x+b
    y.grad = np.ones_like(y.data,dtype=np.float64)
    y._backward()
    x._backward()
    b._backward()
    print(b.grad)
