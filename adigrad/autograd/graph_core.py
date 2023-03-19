"""
Implements the core aspects of the autograd compute
graph.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional

import numpy as np

class Node(ABC):
    """
    A base class representing a Node in our compute graph.
    Forces homogenous connectivity between nodes, and
    requires all sub-classes to implement reverse, allowing
    for really simple backpropogration. 
    """
    def __init__(self) -> None:
        """List of parent and child nodes.
        Multiple are allowed"""
        self.parents : List["Node"] = []
        self.children : List["Node"] = []
    
    @abstractmethod
    def backward(self, *args, **kwargs) -> None:
        """
        An abstract function that all nodes
        need to implement, which backpropogates
        gradients through the network. By forcing
        all nodes to implement this, it becomes
        a little more homogenous.
        """

class Operator(Node, ABC):
    """
    A base class representing some kind of operation.
    Returns either a single tensor or multiple
    tensors. Supports back-propogating gradients
    from operations.

    The init functions should be over-ridden by subclasses
    to store their own needed context there.
    """
    def __init__(self) -> None:
      super().__init__()
    
    """
    A forward function that all operators need to implement.
    Internally, need to apply the operation, set self
    as one of the parents, and configure context
    appropriately
    """
    @abstractmethod
    def forward(self, *args, **kwargs) -> Union["Tensor", List["Tensor"]]:
        pass

    @abstractmethod
    def backward(self) -> None:
        if len(self.children) != 1:
            raise ValueError(f"{len(self.children)} is not a supported number of children for an operator")
    
class Tensor(Node):
    """
    A class representing a Tensor, which is a container
    for data within this library. Wraps around a numpy
    array, and has gradient tracking.
    """
    def __init__(self, data : np.ndarray, parent_op : Optional[Operator] = None, has_grad : bool = True):
        super().__init__()

        self.data : np.ndarray = data
        self.grad : np.ndarray = np.zeros(data.shape)
        if parent_op is not None:
            self.parents.append(parent_op)
        self.has_grad = has_grad
    
    def backward(self, isTerminal : bool) -> None:
        """
        Implements the backward call needed by Node. This doesn't
        actually do much in this case, apart from calling the
        parent Operator and calling backward on it.

        :param isTerminal: A boolean indicating if this is a
          terminal Tensor. These correspond to Tensors storing the
          cost. If yes, self.grad is set to ones first, and then
          backward is called on the parent operator.
        """
        if isTerminal:
            self.grad = np.ones( self.data.shape )
        
        if len(self.parents) == 1:
            self.parents[0].backward()
        elif len(self.parents) != 0:
            raise ValueError(f"Unsupported number of Tensor parents : {len(self.parents)}")
    
    # Defining basic ops on tensors
    def __add__(self, obj2 : "Tensor") -> "Tensor":
        from adigrad.autograd.operators import AddOp
        addOp : AddOp = AddOp()
        return addOp.forward(self, obj2)
    
    def __mul__(self, obj2 : "Tensor") -> "Tensor":
        from adigrad.autograd.operators import ElementwiseMulOp
        mulOp : ElementwiseMulOp = ElementwiseMulOp()
        return mulOp.forward(self, obj2)