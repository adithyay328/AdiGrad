import numpy as np

import adigrad as ag
import adigrad.autograd as ad

a = ad.Tensor(np.array(1))
b = ad.Tensor(np.array(2))

c = a + b

c.backward(True)

print(a.grad)
print(b.grad)
print(c.grad)