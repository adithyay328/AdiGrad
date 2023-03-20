"""
A pytest fixture to test all our operators
"""
import adigrad.autograd as ag

import numpy as np

def test_add_mn():
    """
    Testing our adders on 2 mxn matrices;
    cost should just be ones
    """
    a = ag.Tensor(np.ones((2, 2)))
    b = ag.Tensor(np.ones((2, 2)))

    c = a + b

    c.backward(True)

    assert np.array_equal(a.grad, a.data)
    assert np.array_equal(b.grad, b.data)

def test_add_m_scalar():
    """
    Tests addition of an mxn mat and a scalr
    """
    a = ag.Tensor(np.ones((2, 2)))
    b = ag.Tensor(np.ones((1)))

    c = a + b

    c.backward(True)

    assert np.array_equal(a.grad, a.data)
    assert np.array_equal(np.array([4]), b.grad)

def test_mul_scalar_scalar():
    """
    Tests if the product of 2 scalars is implemented correctly.
    """
    a = ag.Tensor(np.array([5]))
    b = ag.Tensor(np.array([4]))

    c = a * b
    c.backward(True)

    assert np.array_equal(a.grad, b.data)
    assert np.array_equal(b.grad, a.data)

def test_mul_mn_scalar():
    """
    Tests if the elementwise product of a scalar
    and a matrix is implemented correctly.
    """
    a = ag.Tensor( np.ones((2,2)) * 2 )
    b = ag.Tensor( np.array([4]) )

    c = a * b

    c.backward(True)

    assert np.array_equal(a.grad, np.ones( (2, 2) ) * 4)
    assert np.array_equal(b.grad, np.array([8]))

def test_mul_mn_mn():
    """
    Tests if the element-wise product
    of 2 mxn matrices is working as intended
    """
    a = ag.Tensor(
        np.array([
            [2, 3],
            [5, 6]
        ])
    )

    b = ag.Tensor(
        np.array([
            [4, 5],
            [7, 8]
        ])
    )

    c = a * b

    c.backward(True)

    assert np.array_equal( a.grad, b.data )
    assert np.array_equal( b.grad, a.data )

def test_sum_list():
    """
    Tests if the sum operator works
    on a list of tensors
    """
    a = ag.Tensor(
        np.array([1, 0, 0])
    )
    b = ag.Tensor(
        np.array([0, 1, 0])
    )
    c = ag.Tensor(
        np.array([0, 0, 1])
    )

    sum = ag.Tensor.sum([a, b, c])

    sum.backward(True)

    assert np.array_equal( a.grad, np.array([1, 1, 1]) )
    assert np.array_equal( b.grad, np.array([1, 1, 1]) )
    assert np.array_equal( c.grad, np.array([1, 1, 1]) )

def test_sum_tensor():
    """
    Tests if the sum operator works on a single
    tensor
    """
    a = ag.Tensor(
        np.eye( 2, 2 )
    )

    sum = ag.Tensor.sum(a)

    sum.backward(True)

    assert np.array_equal(a.grad, np.ones( (2, 2) ))

def test_exponential_scalar_scalar():
    """
    Tests the exponent operator on 2 scalars
    """
    a = ag.Tensor(
        np.array([5])
    )
    b = ag.Tensor(
        np.array([4]), has_grad=False
    )

    c = a ** b
    c.backward(True)

    assert np.array_equal(a.grad, b.data * (a.data ** (b.data - 1) ))

def test_exponential_mn_scalar():
    """
    Tests the exponential operator on a matrix and a scalar
    """
    a = ag.Tensor(
        np.array([
          [2.0, 4.0],
          [6.0, 7.0]
        ])
    )
    b = ag.Tensor(
        np.array([2.0]), has_grad=False
    )

    c = a ** b
    d = ag.Tensor.sum(c)

    d.backward(True)
    c.backward(False)

    assert np.array_equal(a.grad, a.data * 2)