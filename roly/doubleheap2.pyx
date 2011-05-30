# This file is distributed under the Bottleneck license:
# 
# Copyright (c) 2011 Archipel Asset Management AB
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#       
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
cimport numpy as np
import cython
from numpy cimport PyArray_EMPTY, PyArray_DIMS, PyArray_Copy, NPY_FLOAT64
np.import_array()

__all__ = ['move_median']

cdef extern from "cdoubleheap2.c":
    double Item
    ctypedef struct Mediator:
       double * data  # circular queue of values
       int *  pos   # index into `heap` for each value
       int *  heap  # max/median/min heap holding indexes into `data`.
       int   N     # allocated size.
       int   idx   # position in circular queue
       int   minCt # count of items in min heap
       int   maxCt # count of items in max heap
    Mediator * MediatorNew(int nItems)
    void MediatorInsert(Mediator* m, double v)
    double MediatorMedian(Mediator* m)

@cython.boundscheck(False)
@cython.wraparound(False)
def move_median(np.ndarray[np.float64_t, ndim=1] a, int window):
    """
    Double heap moving window median on 1d float64 numpy array.

    Parameters
    ----------
    a : ndarray
        Imput array
    window : int
        Window length

    Returns
    -------
    y : ndarray
        A moving window median with the same shape as the input array `a`.

    """
    cdef np.npy_intp *dims
    dims = PyArray_DIMS(a)
    cdef int i, n = dims[0]
    if window == 1:
        return PyArray_Copy(a)
    elif window > n:
        raise ValueError("`window` must be less than a.size.")
    elif window <= 0:
        raise ValueError("`window` must be greater than 0.")
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_FLOAT64, 0)
    cdef Mediator * dheap = MediatorNew(window)
    for i in range(window):    
        y[i] = np.nan
    for i in range(window):
        MediatorInsert(dheap, a[i])
    y[window-1] = MediatorMedian(dheap)
    for i in range(window, n):
        MediatorInsert(dheap, a[i])
        y[i] = MediatorMedian(dheap)
    return y
