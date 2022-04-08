from typing import Any

import jax.numpy as np
from gpflow.base import AnyNDArray

from ..clastion import Clastion, InPut, Preprocessor, Put, put


class asarray(Preprocessor):
    def process(self, instance: Clastion, key: Put[Any], value: Any) -> Any:
        return np.asarray(value)


def arrayput(*preprocessors: Preprocessor) -> InPut[AnyNDArray]:
    return put(asarray(), *preprocessors)
