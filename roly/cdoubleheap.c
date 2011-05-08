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

#include <stdlib.h>
#include <stdio.h>
#include "cdoubleheap.h"

enum direction_e {up,down};
typedef enum direction_e direction_e;

static void  presort       (win_s * winstruct);
static void  migrate_small (win_s *winstruct,int index,dnode_s ** heap);
static void  migrate_big   (win_s *winstruct,int index,dnode_s ** heap);
static void  swap_in_heap  (dnode_s ** heap, int loc1, int loc2);
static void  swap_between  (win_s *w,dnode_s ** heap1,dnode_s **heap2,
                            datum_v new_value);


datum_v
update_window(win_s * w, datum_v new_value)
{
    int csw;
    dnode_s *df;

    if(w->nodes[w->index].isbig) {
        df = w->small[0];
        csw = (new_value < df->value) + 2;
    } else {
        df = w->big[0];
        csw = (new_value > df->value);
    }
    switch(csw) {
        case 0:
            df = &(w->nodes[w->index]);
            df->value = new_value;
            migrate_small(w,df->index,w->small);
            break;
        case 1:
            swap_between(w,w->small,w->big,new_value);
            migrate_small(w,w->nodes[w->index].index,w->small);
            migrate_big(w,w->big[0]->index,w->big);
            break;
        case 2:
            df = &(w->nodes[w->index]);
            df->value = new_value;
            migrate_big(w,df->index,w->big);
            break;
        case 3:
            swap_between(w,w->big,w->small,new_value);
            migrate_big(w,w->nodes[w->index].index,w->big);
            migrate_small(w,w->small[0]->index,w->small);
            break;
        default:
            printf("Error, csw = %d. Bye!\n",csw);
            exit(EXIT_FAILURE);
    }
    w->index++;
    if(w->index >= w->nw) w->index = 0;
    return get_median(w);
}

static void 
migrate_small(win_s *w, int index, dnode_s **heap)
{
    direction_e   dir;
    dnode_s     * df;
    int           parent = 0;
    int           child;
    int           left;
    int           right;
    datum_v       value;

    df = heap[index];
    value = df->value;

    if(index==0) dir=down;
    else {
        parent = (index-1)/2;
        dir = heap[parent]->value < df->value ? up : down;
    }
    if (dir == up) while(1) {
            swap_in_heap(heap,index,parent);
            if(parent == 0) break;
            index = parent;
            parent = (index-1)/2;
            if(heap[parent]->value >= value) break;
        } else while(1) {
            left  = 2*index + 1;
            if(left >= w->ns) break;
            right = left  + 1;
            if(right >= w->ns) {
                if(heap[left]->value > value) {
                    swap_in_heap(heap,index,left);   
                }
                break;
            }
            child = heap[left]->value > heap[right]->value? left: right;
            if(heap[child]->value <= value) break;
            swap_in_heap(heap,index,child);
            index = child;
        }
}

static void 
migrate_big(win_s *w, int index, dnode_s **heap)
{
    direction_e   dir;
    dnode_s     * df;
    int           parent = 0;
    int           child;
    int           left;
    int           right;
    datum_v       value;

    df = heap[index];
    value = df->value;

    if(index==0) dir=down;
    else {
        parent = (index-1)/2;
        dir = heap[parent]->value > df->value ? up : down;
    }
    if (dir == up) while(1) {
        swap_in_heap(heap,index,parent);
        if(parent == 0) break;
        index = parent;
        parent = (index-1)/2;
        if(heap[parent]->value <= value) break;
    }
    else while(1) {
        left  = 2*index + 1;
        if(left >= w->nb) break;
        right = left  + 1;
        if(right >= w->nb) {
            if(heap[left]->value < value) {
                swap_in_heap(heap,index,left);   
            }
            break;
        }
        child = heap[left]->value < heap[right]->value? left: right;
        if(heap[child]->value >= value) break;
        swap_in_heap(heap,index,child);
        index = child;
    }
}

static void
swap_in_heap(dnode_s ** heap, int loc1, int loc2)
{
    dnode_s * temp;

    heap[loc1]->index = loc2;
    heap[loc2]->index = loc1;

    temp       = heap[loc1];
    heap[loc1] = heap[loc2];
    heap[loc2] = temp;
}

static void  
swap_between  (win_s *w,dnode_s ** heap1,dnode_s **heap2,datum_v new_value)
{
    dnode_s *node1, *node2;

    node1               = &(w->nodes[w->index]);
    node2               = heap2[0];
    heap1[node1->index] = node2;
    heap2[node2->index] = node1;
    node2->index        = node1->index;
    node1->index        = 0;
    node1->isbig        = !node1->isbig;
    node2->isbig        = !node2->isbig;
    node1->value        = new_value;
}

win_s *
init_winstruct(int nw)
{
    win_s     * w;

    w        = malloc(sizeof *w);
    if (!w) {printf("malloc failed!  Bye."); exit(EXIT_FAILURE);}
    w->nodes = malloc(nw * sizeof *(w->nodes));
    if (!w->nodes) {printf("malloc failed!  Bye."); exit(EXIT_FAILURE);}
    w->small = malloc(nw * sizeof *(w->small));
    if (!w->small) {printf("malloc failed!  Bye."); exit(EXIT_FAILURE);}
    w->index = 0;
    w->nw    = nw;
    w->nb = nw/2; w->ns = nw - w->nb;
    w->big   = w->small + w->ns;
    if (nw % 2 == 1){
        w->odd = 1;
    } else {
        w->odd = 0;
    }
    return w;
}

void        
init_insert(win_s *w, npy_float64 new_value, int idx)
{
    w->nodes[idx].value = new_value;
    w->small[idx] = &(w->nodes[idx]);
}

win_s *
init_presort(win_s *w)
{
    int i;
    presort(w);
    for(i=0;i<w->ns;i++) {
        dnode_s *d;
        d= w->small[i];
        d->index = i;
        d->isbig = 0;
    }
    for(i=0;i<w->nb;i++) {
        dnode_s *d;
        d= w->big[i];
        d->index = i;
        d->isbig = 1;
    }
    return w;
}

void
delete_winstruct(win_s *w)
{
    if (!w) return;
    if (w->nodes) free(w->nodes);
    if (w->small) free(w->small);
    free(w);
}

double
get_median(win_s *w)
{
    if(w->odd) {
        return w->small[0]->value;
    } else {
        return (w->small[0]->value + w->big[0]->value) / 2.0;
    }
}

static int
compar_nodes(const void *first, const void *second)
{
    dnode_s **df,**ds;
    datum_v f,s;

    df = ((dnode_s **)first);
    ds = ((dnode_s **)second);
    f  = (*df)->value;
    s  = (*ds)->value;
    if(f < s) return -1;
    if(f > s) return +1;
    return 0;
}

static void
presort(win_s * w)
{
    int i,j;
    dnode_s * temp;

    qsort(w->small,w->nw,sizeof(w->small[0]),compar_nodes);
    for(i=0,j=(w->ns)-1;i<j;i++,j--) {
        temp        = w->small[i];
        w->small[i] = w->small[j];
        w->small[j] = temp;
    }

}
