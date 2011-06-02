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

cdef extern from "cdoubleheap3.c":
    struct mm_node:
        np.npy_uint32   small
        np.npy_uint64   idx
        np.npy_float64  val
        mm_node         *next
    struct mm_handle:
        int              odd
        np.npy_uint64    n_s
        np.npy_uint64    n_l
        mm_node          **s_heap
        mm_node          **l_heap
        mm_node          **nodes
        mm_node           *node_data
        mm_node           *first
        mm_node           *last
        np.npy_uint64 s_first_leaf
        np.npy_uint64 l_first_leaf
    mm_handle *mm_new(np.npy_uint64 len)
    void mm_insert_init(mm_handle *mm, np.npy_float64 val)
    void mm_update(mm_handle *mm, np.npy_float64 val)
    np.npy_float64 mm_get_median_odd(mm_handle *mm)
    np.npy_float64 mm_get_median_even(mm_handle *mm)
    void mm_free(mm_handle *mm)

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
    cdef np.npy_float64 (*pt2median)(mm_handle *)
    if (window & 1):
        pt2median = &mm_get_median_odd
    else:
        pt2median = &mm_get_median_even
    for i in range(window):    
        y[i] = np.nan
    cdef mm_handle *mm = mm_new(window)
    for i in range(window):
#        print "inserting", i, a[i]
        mm_insert_init(mm, a[i])
    y[window-1] = pt2median(mm)
    for i in range(window, n):
        mm_update(mm, a[i])
        y[i] = pt2median(mm)
    mm_free(mm)
    return y    

