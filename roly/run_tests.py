
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


#time_ll = []
#for window in windows:
#    print(window)
#    time_ll.append(time_med(a, window, linkedlist.move_median))


#time_dh = []
#for window in windows:
#    print(window)
#    time_dh.append(time_fn(doubleheap.move_median, a, window))

# Test against know working version. 
a = np.random.normal(size=10000)
windows = [3, 7, 15, 31, 63, 127, 255]

for window in windows:
    print(3, np.all(median3(a, window)[window:] == 
                    median_slow(a, window)[window:]))
    print(2, np.all(median2(a, window)[window:] == 
                    median_slow(a, window)[window:]))
    print(1, np.all(median1(a, window)[window:] == 
                    median_slow(a, window)[window:]))



# Timing plots. 
a = np.random.normal(size=2000000)
windows = [3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383]
#t_ll  = [time_fn(median_ll, a, window) for window in windows]
t_dh1 = [time_fn(median1, a, window) for window in windows]
t_dh2 = [time_fn(median2, a, window) for window in windows]
t_dh3 = [time_fn(median3, a, window) for window in windows]


#plt.plot(windows, t_ll, 'o-', label='LL')
plt.plot(windows, t_dh1, 'o-', label='DH1')
plt.plot(windows, t_dh2, 'o-', label='DH2')
plt.plot(windows, t_dh3, 'o-', label='DH3')
plt.legend()
plt.show()

