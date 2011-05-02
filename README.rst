====
Roly
====

A comparison of various moving window median algorithms and implementations.

Install
=======

You'll need to install Cython. Then clone or download roly. Then compile::

    roly/roly$ python linkedlist-setup.py build_ext --inplace

Example
=======

Here's how to find the moving window median of a 1d array using the linked
list method::

    >>> import roly
    >>> import numpy as np
    >>> a = np.random.rand(5)
    >>> a
    array([ 0.4645463 ,  0.44380489,  0.31820587,  0.6821211 ,  0.4912904 ])
    >>> roly.linkedlist.move_median(a, 3)
    array([        nan,         nan,  0.44380489,  0.44380489,  0.4912904 ])

roly also has a slow (python for loop) reference inplementation which is
useful for unit testing::

   >>> roly.slow.move_median(a, 3)
   array([        nan,         nan,  0.44380489,  0.44380489,  0.4912904 ])

Roly license
============

Roly contains code from other projects, the license for which should appear
in the corresponding code files.
