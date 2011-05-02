import numpy as np
cimport numpy as np
import cython
from numpy cimport PyArray_EMPTY, PyArray_DIMS, PyArray_Copy, NPY_FLOAT64
np.import_array()

cdef extern from "cmove_median.c":
    struct mm_node
    struct mm_list:
        np.npy_int64 len
        mm_node *head
        mm_node *tail 
        mm_node *min_node
        mm_node *med_node 
    void mm_init_median(mm_list *mm)
    void mm_insert_init(mm_list *mm, np.npy_float64 val)
    void mm_update(mm_list *mm, np.npy_float64 val)
    np.npy_float64 mm_get_median(mm_list *mm)
    void mm_free(mm_list *mm)
    np.npy_float64 mm_get_median(mm_list *mm)
    mm_list mm_new(np.npy_int64 len)

@cython.boundscheck(False)
@cython.wraparound(False)
def move_median(np.ndarray[np.float64_t, ndim=1] a, int window):
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
    cdef mm_list mm = mm_new(window)
    for i in range(window):
        mm_insert_init(cython.address(mm), a[i])
    for i in range(window):    
        y[i] = np.nan
    mm_init_median(cython.address(mm))
    y[window-1] = mm_get_median(cython.address(mm))
    for i in range(window, n):
        mm_update(cython.address(mm), a[i])
        y[i] = mm_get_median(cython.address(mm))
    mm_free(cython.address(mm))
    return y
