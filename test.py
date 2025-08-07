import numpy as np
import torch as t

def make(*arg) -> t.Tensor:
    res = np.random.rand(*arg)
    return t.tensor(res, requires_grad=True)

exit()