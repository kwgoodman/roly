
#!/usr/bin/env python
from __future__ import division

import time
import numpy as np
import matplotlib.pyplot as plt 


def time_fn(fn, *args, **kwargs):
    n_run = 1
    t0 = time.time()
    for i in range(n_run):
        fn(*args, **kwargs)
    t1 = time.time()
    return (t1 - t0) / n_run


from linkedlist import move_median as median_ll
from doubleheap import move_median as median1
from doubleheap2 import move_median as median2
from doubleheap3 import move_median as median3
from slow import move_median as median_slow


if False:
    a = np.random.uniform(size=30)
    print(a)
    median3(a, 11)

# Test against known working version. 
if True:
    a = np.random.normal(size=10000)
    windows = [3, 7, 15, 31, 63, 127, 255]
    
    for window in windows:
        print(3, np.all(median3(a, window)[window:] == 
                        median_slow(a, window)[window:]))
#        print(2, np.all(median2(a, window)[window:] == 
#                        median_slow(a, window)[window:]))
#        print(1, np.all(median1(a, window)[window:] == 
#                        median_slow(a, window)[window:]))



# Timing plots. 
if True:
    num_elements = 2000000
    a = np.random.normal(size=num_elements)
    windows = [3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383,
               32767, 65535]
    #t_ll  = [time_fn(median_ll, a, window) for window in windows]
    t_dh1 = [time_fn(median1, a, window) for window in windows]
    t_dh2 = [time_fn(median2, a, window) for window in windows]
    t_dh3 = [time_fn(median3, a, window) for window in windows]
    
    #plt.plot(windows, t_ll, 'o-', label='LL')
    plt.plot(windows, t_dh1, 'o-', label='DH1')
    plt.plot(windows, t_dh2, 'o-', label='DH2')
    plt.plot(windows, t_dh3, 'o-', label='DH3')
    plt.title('{0} elements'.format(num_elements))
    plt.xlabel('Window size')
    plt.ylabel('Time')
    plt.legend()
    plt.show()

