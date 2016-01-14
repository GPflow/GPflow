import GPflow
import theano
import numpy as np
import unittest

class TransformTests(unittest.TestCase):
    def setUp(self):
        self.x_free = np.random.randn(10)
        self.transforms = [C() for C in GPflow.transforms.Transform.__subclasses__()]

    def test_tf_np_forward(self):
        """ make sure the np forward transforms are the same as the theano ones"""
        x = theano.tensor.dvector()
        ys = [t.tf_forward(x) for t in self.transforms]
        fns = [theano.function([x], y) for y in ys]
        ys_theano = [f(self.x_free) for f in fns]
        ys_np = [t.forward(self.x_free) for t in self.transforms]
        for y1, y2 in zip(ys_theano, ys_np):
            self.failUnless(np.allclose(ys_theano, ys_np))
    
    def test_forward_backward(self):
        ys_np = [t.forward(self.x_free) for t in self.transforms]
        xs_np = [t.backward(y) for t, y in zip(self.transforms, ys_np)]
        for x in xs_np:
            self.failUnless(np.allclose(x, self.x_free))

    def test_logjac(self):
        #theano can't do the jacobian of the identity, so we'll ignore that transform
        transforms = [t for t in self.transforms if not isinstance(t, GPflow.transforms.Identity)]

        x = theano.tensor.dvector()
        jacs = [theano.gradient.jacobian(t.tf_forward(x), x) for t in transforms]
        ld_jacs = [theano.tensor.log(theano.tensor.nlinalg.det(j)) for j in jacs]

        ld_jac_fns = [theano.function([x], j) for j in ld_jacs]
        transform_jacs = [theano.function([x], t.tf_log_jacobian(x)) for t in transforms]
        for f1, f2 in zip(ld_jac_fns, transform_jacs):
            self.failUnless(np.allclose(f1(self.x_free), f2(self.x_free)))

    


if __name__ == "__main__":
    unittest.main()

