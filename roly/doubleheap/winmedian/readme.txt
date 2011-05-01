I.   ABSTRACT

This is a utility for determining the median of a window as it moves 
through a data set.  There are two C files, winmedian.h and winmedian.c.  
There are four entry points in winmedian.c, create_median, update_window, 
delete_median, and get_median.  The prototypes are in winmedian.h.  In 
addition the user must supply a next_value function that returns the next 
datum when the window moves.  

The utility uses an O(log window_size) technique called the "double heap" 
algorithm.  In this scheme the contents of the window is divided into two 
heaps, lower and upper.  The lower heap contains all items <= the median 
and the upper heap contains all items > the median.  Each time the window 
is moved the trailing datum is removed and the new leading datum is added.  
When both belong to the same heap, the new datum replaces the old and is 
sifted into place. If they are in different heaps the new datum becomes 
the new root of "its" heap and the old root value replaces the deleted 
datum.  The two changed values are then sifted into place.

II.  USAGE

The include file contains a typedef for datum_v.  The file sets it to be 
an int.  Change the typedef as needed to match the actual data.

(a) functions supplied by winmedian.c

datum_v update_window    (win_s * winstruct, datum_v new_value);
win_s * create_winstruct (int nw, int pick_small);
void    delete_winstruct (win_s * winstruct);
datum_v get_median       (win_s *winstruct);

The general plan is to create a win_s struct to hold the window and its 
associated data.  Create_winstruct allocates space for the a window 
of size nw, reads the first nw data items and creates the two heaps.
Use get_median to get the initial median.  Thereafter call update_window 
in a loop; it updates the window and returns the updated median.

The user can select whether the median is in the lower heap or the upper 
heap.  This does not matter if the window size is odd; it does if the 
window size is even.

Function delete_median frees space allocated by create_winstruct.

(b) functions to be supplied by the user

datum_v next_value       (void);

Function create_winstruct expects to be able to use this function to 
initialize the window.

III. Performance

The surprising thing is that most of the sifting costs are O(1); the 
exception is that when the root of a heap is replaced sifting is an 
O(log nw).  This happens when the data being deleted is in one 
heap and its replacement belongs in the other.

As of this writing there hasn't been a performance study.  It seems 
probable, however, that a simple O(n) algorithm will be faster for 
small n, e.g., for n ~ 30.

IV.  License

This utility is copyright (c) 2011 by Richard Harter.  The license is a 
modified BSD license.  A copy may be found in this directory in the file 
license.txt.  The license is repeated below:

"Permission is hereby granted, free of charge, to any person        
obtaining a copy of this software and associated documentation     
files (the "Software"), to deal in the Software without            
restriction, including without limitation the rights to use,       
copy, modify, merge, publish, distribute, sublicense, and/or       
sell copies of the Software, and to permit persons to whom the     
Software is furnished to do so, subject to the following           
conditions:                                                        
                                                                   
The above copyright notice and either a copy of this permission    
notice or a reference to where it may be found shall be included   
in in any copy or modification, in whole or in part, of this       
software.                                                          
                                                                   
Derivative works shall include a notice that the software is a     
modified version of the copyrighted software.                      
                                                                   
There is no guarantee that this software is useful for anything    
or that it is any way correct or of value.  The author is not      
responsible for any consequences of using this software."           

