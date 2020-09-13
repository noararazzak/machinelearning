import numpy as np
import random


def practice():
    # How to create an array with ints
    a = np.array([1, 3, 5, 7])
    print(a)

    # Giving the type of object we are creating
    print(type(a))

    # Another way tp create an array with ints

    b = np.arange(2, 20, 4)
    print(b)

    # Giving the type of object we are creating
    print(type(b))

    # How to create an array with floats
    c = np.linspace(3, 6, 6)
    print(c)

    # Giving the type of object we are creating
    print(type(c))

    # Reshaping the array size of c, finding size, shape, dtype, itemsize
    c = c.reshape(2, 3)
    print(c)
    print(c.size)
    print(c.shape)
    print(c.dtype)
    print(c.itemsize)

    # Another way to create an array

    d = np.array([(0.1, 0.4), (3.1, 0.9), (5.6, 4.2)])
    print(d)

    # Testing a boolean property
    print(d < 3)

    # Multiplying or dividing by a number
    e = d*2
    print(e)

    f = d / 1.7
    print(f)

    # Creating an array of zeros
    g = np.zeros((3, 4))
    print(g)

    # Creating a new array by mentioning dtype
    h = np.ones(12, dtype=np.int16)
    print(h)

    # Creating a new array with random inputs and fixing their precision
    i = np.random.random((4, 5))
    print(i)
    np.set_printoptions (precision=2, suppress=True)
    print(i)

    j = np.random.randint(0, 9, 10)
    print(j)
    
    # Use mathematical and statistical notations on a created array
    print(j.sum())

    print(j.max())

    print(j.min())

    print(j.mean())

    print(j.var())

    print(j. std())

    # Use some more interesting properties like summing rows or columns

    k = np.random.randint(0, 10, 12)
    k = k.reshape(3, 4)
    print(k)
    print(k.sum(axis=1))
    print(k.sum (axis=0))
    print(k.std(axis=1))
    print(k.std(axis=0))

    # Use slicing methods
    lil = np.arange(1, 30, 2)

    print(lil)

    l_slice_backwards = lil[-5:]

    print(l_slice_backwards)

    l_slice_forwards = lil[:-5]

    print(l_slice_forwards)
    return



