import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import adigrad as ag
import adigrad.autograd as ad

# a = ad.Tensor(np.array(1))
# b = ad.Tensor(np.array(2))

# c = a + b

# c.backward(True)

# print(a.grad)
# print(b.grad)
# print(c.grad)

# Lin Reg Test
x = np.array([1,0,-1,-2,3])
y = np.array([4, 0, -2, -1.5, 5])

m = np.array([0])
b = np.array([0])

# Make tensors
x = ad.Tensor(x)
y = ad.Tensor(y)

m = ad.Tensor(m)
b = ad.Tensor(b)

LR = 0.001
EPOCHS = 200

for i in range(EPOCHS):
    # Get model prediction
    prod : ad.Tensor = x * m
    yHat : ad.Tensor = prod + b

    # Compute elementwise
    # difference with y
    d : ad.Tensor = yHat + ( ad.Tensor(np.array([-1]), None, True ) * y )

    # Sum of squares
    dSquared : ad.Tensor = d ** ad.Tensor( np.array([2]), None, False )

    cost = ad.Tensor.sum( dSquared )

    cg = ad.ComputeGraph()
    cg.buildGraph(cost)

    cg.backprop()

    m.data = m.data - m.grad * LR
    b.data = b.data - b.grad * LR

    print(m.data)
    print(b.data)

    cg.zero_grad()