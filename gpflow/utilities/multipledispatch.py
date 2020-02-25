from multipledispatch import Dispatcher as GeneratorDispatcher
from multipledispatch.dispatcher import variadic_signature_matches
from multipledispatch.variadic import isvariadic

__all__ = ["Dispatcher"]


class Dispatcher(GeneratorDispatcher):
    """
    multipledispatch.Dispatcher uses a generator to yield the 
    desired function implementation, which is problematic as TensorFlow's
    autograph is not able to compile code that passes through generators.

    This class overwrites the problematic method in the original
    Dispatcher and solely makes use of simple for-loops, which are
    compilable by AutoGraph.
    """

    def dispatch(self, *types):
        """
        Returns matching function for `types`; if not existing returns None.
        """
        if types in self.funcs:
            return self.funcs[types]

        return self.get_first_occurrence(*types)

    def get_first_occurrence(self, *types):
        """ 
        Returns the first occurrence of a matching function 
        
        Based on `multipledispatch.Dispatcher.dispatch_iter`, which
        returns an iterator of matching functions. This method uses
        the same logic to select functions, but simply returns the first
        element of the iterator. If no matching functions are found, 
        `None` is returned.
        """
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):
                result = self.funcs[signature]
                return result
            elif len(signature) and isvariadic(signature[-1]):
                if variadic_signature_matches(types, signature):
                    result = self.funcs[signature]
                    return result
        return None
