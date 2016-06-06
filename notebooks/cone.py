import numpy as np
import tensorflow as tf

import GPflow
from GPflow.kernels import Kern
from GPflow.param import Param 
from IPython import embed

class ConeKernel(Kern): 
   #RBF kernel on the surface of a cone.
   #I'm not aware of a use for this other than to prove a point...
   #...unless of course you have data that are constrained to be on a cone!
   
   #Let coordinates be 
   #\phi \in [0,2\pi) which denotes the angle on the cone.
   #\r  which denotes the distance along the cone.
   #\theta \in [0,1/2 \pi) which denotes the angle of the cone and is a hyperparameter.
   
   #This is just equivalent to spherical polars where \theta fixed enforces the cone property.
   #Therefore 
   #x = \r sin \theta cos \phi
   #y = \r sin \theta sin \phi
   #z = \r cos \theta
   
   def __init__(self, variance=1.0, lengthscales=None, theta=1.):
      self.rbfKernel = GPflow.kernels.RBF(input_dim=3, variance=variance, lengthscales=lengthscales)
      self.theta = Param(theta)
      
   def Kdiag(self, X):
      zeros = X[:, 0] * 0
      return zeros + self.rbfKernel.variance

   def getCartesianCoordinates(self, phi, r ):
      x = tf.expand_dims( r*tf.sin(self.theta) * tf.cos(phi), 1 )
      y = tf.expand_dims( r*tf.sin(self.theta) * tf.sin(phi), 1 )
      z = tf.expand_dims( r*tf.cos(self.theta), 1 )
      return tf.concat(1, [x,y,z] )

   def K(self, X, X2=None):
      phiA = X[:,0]
      rA = X[:,1]
      XTransformed= self.getCartesianCoordinates(phiA, rA)
      if X2!=None:
         phiB = X2[:,0]
         rB = X2[:,1]
         X2Transformed= self.getCartesianCoordinates(phiB, rB)
      else:
         X2Transformed = None
      return self.rbfKernel.K( XTransformed, X2Transformed )
      
def coneRegression():
   nDataPoints = 100 
   noiseSd = 0.05
   phis = np.linspace( 0.,np.pi*2., nDataPoints )
   rs = np.ones( nDataPoints ) 
   X = np.vstack( [phis, rs] ).T
   Y = np.atleast_2d(np.sin( phis ) + np.random.randn( *phis.shape ) * noiseSd).T
   Z = np.vstack( [np.linspace( 0., np.pi*2., 10 ), np.ones(10) ] ).T
   embed()
   model = GPflow.sgpr.SGPR( X=X, Y=Y, kern =  ConeKernel() , Z=Z )
   model.optimize()    
   embed()

if __name__ == "__main__":
   coneRegression()
