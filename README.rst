====
Roly
====

A comparison of various moving window median algorithms and implementations.

Roly current contains three moving window median algorithms:

- Python "for loop"
- Linked list written in C and wrapped in Cython
- Double heap (3 implementations) written in C and wrapped in Cython

So far all the moving window functions in roly calculate the median of a 1d
window, not 2d is often done in image work.

The roly project is discussed on the Bottleneck mailing list:
http://groups.google.com/group/bottle-neck

Install
=======

You'll need to install Cython. Then clone or download roly. Then make::

    $ cd roly/roly
    $ make all

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

The are three implementations of the double heap method (the first one
contains a bug)::

    >>> roly.doubleheap.move_median
    >>> roly,doubleheap2.move_median
    >>> roly.doubleheap3.move_median

roly also has a slow (python for loop) reference inplementation which is
useful for unit testing::

   >>> roly.slow.move_median(a, 3)
   array([        nan,         nan,  0.44380489,  0.44380489,  0.4912904 ])

Performance
===========

Roly contain a benchmark. To run it::

    $ cd roly/roly
    $ python run_tests.py

Roly license
============

Roly contains code from other projects, the license for which should appear
in the corresponding code files.
