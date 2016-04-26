import tensorflow as tf
import numpy as np
from GPflow.param import Param, Parameterized, ParamList

class MeanFunction(Parameterized):
    """
    The base mean function class.
    To implement a mean funcion, write the __call__ method. This takes a 
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X. 
    MeanFunction classes can have parameters, see the Linear class for an example.
    """
    def __call__(self, X):
        raise NotImplementedError, "Implement the __call__ method for this mean function"
    
    def __add__(self, other):
        raise NotImplementedError, "Implement the __add__ method for this mean function"
        

    def __mul__(self, other):
        raise NotImplementedError, "Implement the __mul__ method for this mean function"
        

class Zero(MeanFunction):
    def __call__(self, X):
        return tf.zeros(tf.pack([tf.shape(X)[0], 1]), dtype='float64')
        
    def __add__(self, other):
        return other
    def __mul__(self, other):
        return self   


class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """
    def __init__(self, A=np.ones((1,1)), b=np.zeros(1)):
        """
        A is a matrix which maps each element of X to Y, b is an additive constant.
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q. 
        """
        MeanFunction.__init__(self)
        self.A = Param(np.atleast_2d(A))
        self.b = Param(b)
    def __call__(self, X):
        return tf.matmul(X, self.A) + self.b
        
    def __add__(self, other):  
        if isinstance(other, Zero):
            return self
        elif isinstance(other, Constant):
            return Linear(other.A, other.b + self.c)
        elif isinstance(other, Linear):
            return Linear(self.A + other.A, self.b + other.b)
        elif isinstance(other, PolyAdd):
            return PolyAdd(self, other)
        elif isinstance(other, PolyProd):
            return PolyAdd(self, other)
            
    def __mul__(self, other):
        if isinstance(other, Zero):
            return other
        elif isinstance(other, Constant):
            return Linear(other.A, other.b * self.c)
        elif isinstance(other, Linear):
            return PolyProd(self, other)
        elif isinstance(other, PolyAdd):
            return PolyProd(self, other)
        elif isinstance(other, PolyProd):
            return PolyProd(self, other)

class Constant(MeanFunction):
    """
    y_i = c,,
    """
    def __init__(self, c=np.zeros(1)):
        MeanFunction.__init__(self)
        self.c = Param(c)
    def __call__(self, X):
        return tf.tile(tf.reshape(self.c, (1,-1)), tf.pack([tf.shape(X)[0], 1])) 
    def __add__(self, other):  
        if isinstance(other, Zero):
            return self
        elif isinstance(other, Constant):
            return Constant(self.c + other.c)
        elif isinstance(other, Linear):
            return Linear(other.A, other.b + self.c)
        elif isinstance(other, PolyAdd):
            return PolyAdd(self, other)
        elif isinstance(other, PolyProd):
            return PolyAdd(self, other)
            
    def __mul__(self, other):
        if isinstance(other, Zero):
            return other
        elif isinstance(other, Constant):
            return Constant(self.c * other.c)
        elif isinstance(other, Linear):
            return Linear(other.A, other.b * self.c)
        elif isinstance(other, PolyAdd):
            PolyProd(self, other)
        elif isinstance(other, PolyProd):
            PolyProd(self, other)

class PolyAdd(MeanFunction):
    def __init__(self, mfunA, mfunB):
        MeanFunction.__init__(self)
        self.mfunA = mfunA
        self.mfunB = mfunB
        
    def __call__(self, X):
        return tf.add(self.funA(X), self.funB(X))
        
    def __add__(self, other):
        if isinstance(other, Zero):
            return self
        else: return PolyAdd(self, other)
        
    def __mul__(self, other):
        if isinstance(other, Zero):
            return other
        else: return PolyProd(self, other)
                    
class PolyProd(MeanFunction):
    def __init__(self, mfunA, mfunB):
        MeanFunction.__init__(self)
        self.mfunA = mfunA
        self.mfunB = mfunB
    def __call__(self, X):
        return tf.matmul(self.mfunA(X), self.mfunB(X))
    def __add__(self, other):
        if isinstance(other, Zero):
            return self
        else: return PolyAdd(self, other)        
    def __mul__(self, other):
        if isinstance(other, Zero):
            return other
        else: return PolyProd(self, other)