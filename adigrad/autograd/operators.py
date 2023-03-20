"""
This module implements all the operators
that are built into this library
"""
from typing import Any, List, Union

import numpy as np

from adigrad.autograd.graph_core import Tensor, Operator, Node

class AddOp(Operator):
    """
    Implements generalized(scalar, vector, matrix or general shaped tensor) addition,
    and its derivative.
    """
    def forward(self, t1 : Tensor, t2 : Tensor) -> Tensor:
        # As of now, we only support some addition shapes
        isShapeValid = (
            t1.data.shape == t2.data.shape
            or t1.data.shape == (1,)
            or t2.data.shape == (1,)
        )

        if not isShapeValid:
            raise ValueError(f"Invalid addition shape: {t1.data.shape} + {t2.data.shape}")
        else:
            newT : Tensor = Tensor(t1.data + t2.data, self, True)

            self.parents += [t1, t2]
            self.children.append(newT)

            return newT
    
    def backward(self) -> None:
        """
        Backward has 2 main cases on the AddOp; one where
        Tensors have same size, other where one is a scalar
        """
        super().backward()

        parentError = lambda: TypeError("Expected all parents of an Operator to be a tensor")

        assert len(self.parents) == 2
        assert len(self.children) == 1
        
        if type(self.children[0]) is not Tensor:
            raise parentError()
        
        childGrad : np.ndarray = self.children[0].grad

        # 2 cases we deal with: equal matrix shapes,
        # or one where one is a scalar
        parentOne : Any = self.parents[0]
        parentTwo : Any = self.parents[1]

        assert type(parentOne) == Tensor
        assert type(parentTwo) == Tensor

        if parentOne.data.shape == parentTwo.data.shape or parentOne.data.shape == (1,) or parentTwo.data.shape == (1,):
            parentOne.grad += childGrad if childGrad.shape == parentOne.grad.shape else np.sum(childGrad)
            parentTwo.grad += childGrad if childGrad.shape == parentTwo.grad.shape else np.sum(childGrad)
        else:
            raise ValueError("Illegal parent shape.")

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
        super().backward()

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
            # This only runs if the shape is valid
            
            # First compute jacobians, assuming both are equal size matrices.
            t1Grad = child.grad * t2.data
            t2Grad = child.grad * t1.data

            # Now, we need to add to their grad. If the shape is a scalar,
            # it has an impact on all the entries of the other matrix,
            # so sum over all entries of grad
            t1.grad += t1Grad if t1Grad.shape == t1.grad.shape else np.sum(t1Grad)
            t2.grad += t2Grad if t2Grad.shape == t2.grad.shape else np.sum(t2Grad)
        else:
            raise ValueError(f"Unsupported elementwise product between tensors of size {t1.data.shape} , {t2.data.shape}")


class SumOp(Operator):
    """
    An operator implementing forward and backward
    passes through a tensor sum. 
    
    In the case of a list of Tensors it sums them up
    and retains the same shape on the output(
      all input tensors need to be the same shape
    )

    In the case of a single tensor, it sums up
    all the elements in that and returns a scalar.
    """
    def __init__(self):
        super().__init__()

        self.wasInputAList : bool = False

    def forward(self, tensorIn : Union[List[Tensor], Tensor]) -> Tensor:
        """
        Computes the sum over all input tensors.

        :param tensor: Either a list of Tensors or a
          single tensor to sum over.
        """
        if type(tensorIn) == list:
            self.wasInputAList = True

            # Confirm shape is uniform
            firstShape = tensorIn[0].data.shape
            for tensor in tensorIn[1:]:
                if tensor.data.shape != firstShape:
                    raise ValueError("Can't sum over tensors with different shapes.")

            newT : Tensor = Tensor(
                np.sum( np.vstack( [tensor.data for tensor in tensorIn] ), axis=0 ),
                self, True
            )

            self.parents += tensorIn
            self.children.append(newT)

            return newT
        elif type(tensorIn) == Tensor:
            self.wasInputAList = False

            newT : Tensor = Tensor(
                np.sum(tensorIn.data), self, True
            )

            self.parents.append(tensorIn)
            self.children.append(newT)

            return newT
        else:
            raise ValueError("Invalid input type of tensorIn.")

    def backward(self) -> None:
        """
        Implements backwards over all input
        tensors.
        """
        assert len(self.children) == 1 and type(self.children[0]) == Tensor
        childGrad : np.ndarray = self.children[0].grad

        if self.wasInputAList:
            # If input was a list, then just pass back
            # jacobian as is
            for parent in self.parents:
                assert type(parent) == Tensor
                parent.grad += childGrad
        else:
            parent = self.parents[0]
            assert type(parent) == Tensor
            parent.grad += np.ones( parent.data.shape ) * childGrad

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
            data = inT.data ** power.data, parent_op=self, has_grad=True
        )

        self.parents += [inT, power]
        self.children.append(newT)

        return newT
    
    def backward(self) -> None:
        """
        When computing derivatives in this case, we will not
        compute the derivative on the exponent; leads to a lot of
        errors
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
        assert power.has_grad == False

        # Computing gradients; power is a bit interesting, since we need
        # to first compute the jacobian
        inT.grad += power.data[0] * np.power(inT.data, power.data[0] - 1) * childGrad