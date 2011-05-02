/* --------------------------------------------------------------------- */
/*                 BEGIN COPYRIGHT AND LICENSE NOTICE                    */
/* --------------------------------------------------------------------- */
/*
** Copyright (c) 2011 by Richard Harter.                                
**                                                                      
** Permission is hereby granted, free of charge, to any person          
** obtaining a copy of this software and associated documentation       
** files (the "Software"), to deal in the Software without              
** restriction, including without limitation the rights to use,         
** copy, modify, merge, publish, distribute, sublicense, and/or         
** sell copies of the Software, and to permit persons to whom the       
** Software is furnished to do so, subject to the following             
** conditions:                                                          
**                                                                      
** The above copyright notice and either a copy of this permission 
** notice or a reference to where it may be found shall be included 
** in in any copy or modification, in whole or in part, of this 
** software.  
**                                                                      
** Derivative works shall include a notice that the software is a       
** modified version of the copyrighted software.                        
**                                                                      
** There is no guarantee that this software is useful for anything      
** or that it is any way correct or of value.  The author is not        
** responsible for any consequences of using this software.             
**
*/
/* --------------------------------------------------------------------- */
/*                 END COPYRIGHT AND LICENSE NOTICE                      */
/* --------------------------------------------------------------------- */

/* 2001 Modified by Keith Goodman for bottleneck */

#ifndef HAVE_WINMEDIAN_H
#define HAVE_WINMEDIAN_H

/* ------------------------ typedefs ----------------------- */

typedef struct win_s   win_s;
typedef struct dnode_s dnode_s;
typedef double    datum_v;

/* ---------------- struct definitions --------------------- */

struct dnode_s {
    datum_v value;
    int     index;
    int     isbig;
};

struct win_s {
    dnode_s          * nodes;
    dnode_s         ** small;
    dnode_s         ** big;
    int                nw;
    int                ns;
    int                nb;
    int                index;
};

/* --------------------- prototypes ------------------------ */

datum_v update_window    (win_s * winstruct, datum_v new_value);
win_s * create_winstruct (int nw, double * a);
void    delete_winstruct (win_s * winstruct);
datum_v next_value       (void);
datum_v get_median       (win_s *winstruct);

#endif
