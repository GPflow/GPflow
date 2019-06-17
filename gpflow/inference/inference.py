from abc import abstractmethod


class Variational:
    @abstractmethod
    def elbo(self, *args, **kwargs):
        pass