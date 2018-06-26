class Likelihood:
    def predict_density(self):
        self.logp()
        print("quadrature")

class Gaussian(Likelihood):
    def predict_density(self):
        print("analytic")

    def logp(self):
        print("logp")

class MC(Likelihood):
    def predict_density(self):
        self.logp()
        print("mc")

class GaussianMC(Gaussian, MC):
    pass

class MCGaussian(MC, Gaussian):
    pass

