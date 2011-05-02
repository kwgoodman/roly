import numpy as np
cimport numpy as np
import cython

# Linked list code from:
# http://pages.cs.wisc.edu/~johnl/median_code/median_code.c
# Put the code in a file names cmove_median.c in the same dir as this file
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
    cdef int n = a.size, i
    if window > n:
        raise ValueError("`window` must be less than a.size.")
    if window < 2:
        raise ValueError("I get a segfault when `window` is 1.")
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(n) 
    cdef mm_list mm = mm_new(window)
    for i in range(window):
        mm_insert_init(cython.address(mm), a[i])
        y[i] = np.nan
    mm_init_median(cython.address(mm))
    y[window-1] = mm_get_median(cython.address(mm))
    for i in range(window, n):
        mm_update(cython.address(mm), a[i])
        y[i] = mm_get_median(cython.address(mm))
    mm_free(cython.address(mm))
    return y
