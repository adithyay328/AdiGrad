"""
This module implements all the operators
that are built into this library
"""
from typing import Any

import numpy as np

from adigrad.autograd.graph_core import Tensor, Operator, Node

class AddOp(Operator):
    """
    Implements generalized(scalar, vector, matrix or general shaped tensor) addition,
    and its derivative.
    """
    def forward(self, t1 : Tensor, t2 : Tensor) -> Tensor:
        newT : Tensor = Tensor(t1.data + t2.data, self, True)

        self.parents += [t1, t2]
        self.children.append(newT)

        return newT
    
    def backward(self) -> None:
        """
        Backward is really simple on AddOps; just return ones
        of the same shape as the current grad of all child
        tensors, and multiply with parent grad; this can
        be done elementwise, since shape is consistent
        across this operation
        """
        super().backward()

        parentError = lambda: TypeError("Expected all parents of an Operator to be a tensor")

        assert len(self.parents) == 2
        assert len(self.children) == 1
        
        if type(self.children[0]) is not Tensor:
            raise parentError()
        
        childGrad : np.ndarray = self.children[0].grad

        for parent in self.parents:
            if type(parent) is not Tensor:
                raise parentError()
            
            parent.grad += np.ones(parent.grad.shape) * childGrad
        
        for parent in self.parents:
            if type(parent) is not Tensor:
                raise parentError()
            parent.backward(False)

class ElementwiseMulOp(Operator):
    """
    Implements elementwise multiplication
    of 2 tensors, and its back-propogation.

    Can deal with different shapes, based on
    what size of tensors
    """

    def forward(self, t1 : Tensor, t2 : Tensor) -> Tensor:
        """
        :param t1: The first tensor to elementwise multiply.
        :param t2: The second tensor to elementwise multiply.

        :return: The output tensor, with gradient tracking handled.
        """
        # I haven't studied the entire range of supported numpy
        # multiply size, so for now we will stick to a known
        # subset that we backprop for
        isSupportedSize = (
            t1.data.shape == t2.data.shape or 
            t1.data.shape == (1,) or t2.data.shape == (1,)
        )

        # Makes sure we have a supported size of input tensors
        if isSupportedSize:
            newT : Tensor = Tensor(data = t1.data * t2.data, parent_op=self, has_grad=True)
            
            self.children.append(newT)
            self.parents += [t1, t2]

            return newT
        else:
            raise ValueError(f"Unsupported elementwise product between tensors of size {t1.data.shape} , {t2.data.shape}")
    
    def backward(self) -> None:
        """
        Implements the backward pass.
        """
        for parent in self.parents:
            if type(parent) is not Tensor:
                raise ValueError("Parent not of the right type.")
            
        assert len(self.parents) == 2

        t1 : Node = self.parents[0]
        t2 : Node = self.parents[1]

        child : Node = self.children[0]

        assert type(t1) is Tensor 
        assert type(t2) is Tensor
        assert type(child) is Tensor

        if t1.data.shape == t2.data.shape or t1.data.shape == (1,) or t2.data.shape == t2.data.shape == (1,):
            # Same logic as element-wise product rule
            t1.grad += child.grad * t2
            t2.grad += child.grad * t1

class SumOp(Operator):
    """
    An operator implementing forward and backward
    passes through sum operations.
    """
    def forward(self, *tensors) -> "Tensor":
        """
        Computes the sum over all input tensors.

        :param *tensors: A list of tuples passed in
            that we will sum over
        """
        newT : Tensor = Tensor(
            np.sum( tensors ), self, True
        )

        self.parents += list(tensors)
        self.children.append(newT)

        return newT

    def backward(self) -> None:
        """
        Implements backwards over all input
        tensors.
        """
        assert len(self.children) == 1 and type(self.children[0]) is Tensor

        childGrad : np.ndarray = self.children[0].grad

        for parent in self.parents:
            assert type(parent) == Tensor

            parent.grad += np.ones(parent.grad.shape) * childGrad

class PowerOp(Operator):
    """
    An operator implementing forward and backward
    passes through an elementwise 
    operator of the form x^y; both x and y can
    be variables.
    """
    def forward(self, inT : Tensor, power : Tensor) -> "Tensor":
        """
        The forward function for this operator.
        inT can be of any shape, but power can only be a scalar
        tensor for now.

        :param inT: The tensor to raise to a power.
        :param power: A tensor storing the power to raise
          inT to.

        :return: The resultant Tensor.
        """
        assert power.data.shape == (1,)

        # Compute elementwise power
        newT : Tensor = Tensor(
            data = inT.data ** power, parent_op=self, has_grad=True
        )

        self.parents += [inT, power]
        self.children.append(newT)

        return newT
    
    def backward(self) -> None:
        """
        When computing derivatives in this case, we have to effectively
        use 2 different derivative rules:
        #. Power Rule for inT's elements
        #. Exponent rule for the power tensor
        """
        assert len(self.parents) == 2
        assert len(self.children) == 1

        assert type(self.children[0]) is Tensor

        childGrad : np.ndarray = self.children[0].grad

        inT : Any = self.parents[0]
        power : Any = self.parents[1]
        assert type(inT) == type(power) == Tensor

        # Computing gradients; power is a bit interesting, since we need
        # to first compute the jacobian
        inT.grad += power.data[0] * np.power(inT.data, power.data[0] - 1) * childGrad
        
        # Stores a matrix of partial derivatives, where each element is the partial
        # of the corresponding resultant matrix's element w.r.t power.
        # Then just sum up product of this and childGrad
        elementwiseJacobian = np.power(inT, power) * np.log(power[0])
        jacobianWrtCost = elementwiseJacobian * childGrad

        # Sum up all partials, and assign that to the grad of power
        power.grad += np.sum(jacobianWrtCost)