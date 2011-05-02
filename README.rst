====
Roly
====

A comparison of various moving window median algorithms and implementations.

Roly current contains three moving window median algorithms:

- Python "for loop"
- Linked list written in C and wrapped in Cython
- Double heap written in C and wrapped in Cython

Install
=======

You'll need to install Cython. Then clone or download roly. Then compile::

    roly/roly$ python linkedlist-setup.py build_ext --inplace

Example
=======

Let's find the moving window median of a 1d array using the linked list
and double heap methods. First create a numpy array::

    >>> import roly
    >>> import numpy as np
    >>> a = np.random.rand(5)
    >>> a
    array([ 0.4645463 ,  0.44380489,  0.31820587,  0.6821211 ,  0.4912904 ])

Then try the linked list method::

    >>> roly.linkedlist.move_median(a, 3)
    array([        nan,         nan,  0.44380489,  0.44380489,  0.4912904 ])

And the double heap method::

    >>> roly.doubleheap.move_median(a, 3)
    array([        nan,         nan,  0.44380489,  0.44380489,  0.4912904 ])

roly also has a slow (python for loop) reference inplementation which is
useful for unit testing::

   >>> roly.slow.move_median(a, 3)
   array([        nan,         nan,  0.44380489,  0.44380489,  0.4912904 ])

Performance
===========

A comparison of the performance for the linked list and the double heap
algorithm for various window sizes::

    >>> a = np.random.rand(1e5)

    >>> window = 10
    >>> timeit roly.slow.move_median(a, window)
    1 loops, best of 3: 2.57 s per loop
    >>> timeit roly.linkedlist.move_median(a, window)
    100 loops, best of 3: 4.57 ms per loop
    >>> timeit roly.doubleheap.move_median(a, window)
    100 loops, best of 3: 4.87 ms per loop

    >>> window = 100
    >>> timeit roly.linkedlist.move_median(a, window)
    10 loops, best of 3: 19.4 ms per loop
    >>> timeit roly.doubleheap.move_median(a, window)
    100 loops, best of 3: 6.55 ms per loop

    >>> window = 1000
    >>> timeit roly.linkedlist.move_median(a, window)
    1 loops, best of 3: 206 ms per loop
    >>> timeit roly.doubleheap.move_median(a, window)
    100 loops, best of 3: 7.76 ms per loop

    >>> window = 10000
    >>> timeit roly.linkedlist.move_median(a, window)
    1 loops, best of 3: 4.56 s per loop
    >>> timeit roly.doubleheap.move_median(a, window)
    100 loops, best of 3: 10.2 ms per loop

The double heap is much faster than the linked list except at small window
widths. And even then it is not far behind.

Roly license
============

Roly contains code from other projects, the license for which should appear
in the corresponding code files.
