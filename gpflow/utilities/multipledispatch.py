from multipledispatch import Dispatcher as GeneratorDispatcher
from multipledispatch.dispatcher import variadic_signature_matches
from multipledispatch.variadic import isvariadic


class Dispatcher(GeneratorDispatcher):
    """
    Multipledispatch's Dispatcher uses a generator to yield the 
    desired function implmentation, which is problematic as TF
    autograph is not able to compile code that passes through generators.

    This class overwrites the problematic method in the original
    Dispatcher and makes soley use of simple for-loops, which are
    compilable by AutoGraph.
    """

    def dispatch(self, *types):
        """
        Returns matching function for `types`, if not existins returns None.
        """
        if types in self.funcs:
            return self.funcs[types]

        return self.get_first_occurence(*types)

    def get_first_occurence(self, *types):
        """ Returns the first occurrence of a mathcing function """
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
