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

cdef extern from "cdoubleheap.c":
    double datum_v
    struct dnode_s:
        double  value # Should be type datum_v but cython doesn't like that
        int     index
        int     isbig
    struct win_s:
        dnode_s  * nodes
        dnode_s ** small
        dnode_s ** big
        int        nw
        int        ns
        int        nb
        int        index
        int        odd
    win_s * init_winstruct(int nw)
    void init_insert(win_s *w, double new_value, int idx)
    win_s* init_presort(win_s *w)
    float get_median(win_s *w)
    # Both double in next line should be datum_v but cython complains
    double update_window(win_s * w, double new_value)
    void delete_winstruct(win_s *w)

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
    for i in range(window):    
        y[i] = np.nan
    cdef win_s * dheap = init_winstruct(window)
    for i in range(window):
        init_insert(dheap, a[i], i)
    dheap = init_presort(dheap)
    y[window-1] = get_median(dheap)
    for i in range(window, n):
        y[i] = update_window(dheap, a[i])
    delete_winstruct(dheap)
    return y
