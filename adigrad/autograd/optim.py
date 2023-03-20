"""
Contains logic for all actual optimization
"""
from typing import Set, List

import networkx as nx
import numpy as np

import adigrad as ad
from adigrad import autograd as ag

class ComputeGraph:
    """
    A class representing a compute graph.
    We don't build it in the forward pass,
    rather we just set up all parent
    connections. This class goes
    backwards through all connections
    and builds a networkx graph.
    """
    def __init__(self):
        self.graph : nx.DiGraph = nx.DiGraph()
    
    def buildGraph(self, endTensor : ag.Tensor) -> None:
        """
        Walks backwards from a tensor at the end of a compute
        graph(usually a cost tensor) and builds the compute
        graph and saves internally.

        :param endTensor: The tensor to build backwards from.
        """
        
        # Keeping track of the leaves in the graph
        # here. When this is empty, we're done
        count = 0

        leaves : Set[ag.Node] = set([endTensor])

        while len(leaves) > 0:
            currLeaves : List[ag.Node] = list(leaves)
            leaves.clear()

            # BFS backwards
            for leaf in currLeaves:
                # Add all parents to the nx graph
                for parent in leaf.parents:
                    self.graph.add_node( parent )

                # Build edges for each parent
                self.graph.add_edges_from(
                    [ (parent, leaf) for parent in leaf.parents ]
                )

                # Add all parents to leaves
                leaves = leaves | set(leaf.parents)
        
    def zero_grad(self) -> None:
        """
        Iterates over all the Tensors in this compute graph,
        and zeros their grad and clears their parents.
        """
        for node in self.graph.nodes:
            if type(node) == ag.Tensor:
                node.grad = np.zeros(
                    node.data.shape
                )
            
            node.children = []
            node.parents = []
    
    def backprop(self) -> None:
        """
        Performs a reverse topological sort,
        and computes all gradients.
        """
        # print("Started")
        reverseTopo = list(reversed(list((nx.topological_sort(self.graph)))))

        # First node needs to be ones
        assert issubclass( type(reverseTopo[0]) , ag.Node)
        reverseTopo[0].backward(True)

        for node in reverseTopo[1:]:
            if type(node) == ag.Tensor:
              if node.has_grad:
                node.backward(False)
            elif issubclass(type(node), ag.Operator):
              node.backward()
            else:
                raise ValueError(f"Invalid node type : {type(node)}")