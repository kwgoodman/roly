//Copyright (c) 2011 ashelly.myopenid.com under <http://www.opensource.org/licenses/mit-license>

#include <stdlib.h>
#define inline

typedef double Item;
typedef struct Mediator_t
{
   Item* data;  //circular queue of values
   int*  pos;   //index into `heap` for each value
   int*  heap;  //max/median/min heap holding indexes into `data`.
   int   N;     //allocated size.
   int   idx;   //position in circular queue
   int   minCt; //count of items in min heap
   int   maxCt; //count of items in max heap
   int   odd;
} Mediator;

/*--- Helper Functions ---*/

//returns 1 if heap[i] < heap[j]
inline int mmless(Mediator* m, int i, int j)
{
   return (m->data[m->heap[i]] < m->data[m->heap[j]]);
}

//swaps items i&j in heap, maintains indexes
int mmexchange(Mediator* m, int i, int j)
{
   int t = m->heap[i];
   m->heap[i]=m->heap[j];
   m->heap[j]=t;
   m->pos[m->heap[i]]=i;
   m->pos[m->heap[j]]=j;
   return 1;
}

//swaps items i&j if i<j;  returns true if swapped
inline int mmCmpExch(Mediator* m, int i, int j)
{
   return (mmless(m,i,j) && mmexchange(m,i,j));
}

//maintains minheap property for all items below i.
void minSortDown(Mediator* m, int i)
{
   for (i*=2; i <= m->minCt; i*=2)
   {  if (i < m->minCt && mmless(m, i+1, i)) { ++i; }
      if (!mmCmpExch(m,i,i/2)) { break; }
   }
}

//maintains maxheap property for all items below i. (negative indexes)
void maxSortDown(Mediator* m, int i)
{
   for (i*=2; i >= -m->maxCt; i*=2)
   {  if (i > -m->maxCt && mmless(m, i, i-1)) { --i; }
      if (!mmCmpExch(m,i/2,i)) { break; }
   }
}

//maintains minheap property for all items above i, including median
//returns true if median changed
inline int minSortUp(Mediator* m, int i)
{
   while (i>0 && mmCmpExch(m,i,i/2)) i/=2;
   return (i==0);
}

//maintains maxheap property for all items above i, including median
//returns true if median changed
inline int maxSortUp(Mediator* m, int i)
{
   while (i<0 && mmCmpExch(m,i/2,i))  i/=2;
   return (i==0);
}

/*--- Public Interface ---*/

//creates new Mediator: to calculate `nItems` running median. 
//mallocs single block of memory, caller must free.
Mediator* MediatorNew(int nItems)
{
   int size = sizeof(Mediator)+nItems*(sizeof(Item)+sizeof(int)*2);
   Mediator* m=  malloc(size);
   m->data= (Item*)(m+1);
   m->pos = (int*) (m->data+nItems);
   m->heap = m->pos+nItems + (nItems/2); //points to middle of storage.
   m->N=nItems;
   m->minCt = m->maxCt = m->idx = 0;
   while (nItems--)  //set up initial heap fill pattern: median,max,min,max,...
   {  m->pos[nItems]= ((nItems+1)/2) * ((nItems&1)?-1:1);
      m->heap[m->pos[nItems]]=nItems;
   }
   if (nItems % 2 == 1){
      m->odd = 1;
   } else {
      m->odd = 0;
   }
   return m;
}

//Inserts item, maintains median in O(lg nItems)
void MediatorInsert(Mediator* m, Item v)
{
   int p = m->pos[m->idx];
   Item old = m->data[m->idx];
   m->data[m->idx]=v;
   m->idx = (m->idx+1) % m->N;
   if (p>0)         //new item is in minHeap
   {  if (m->minCt < (m->N-1)/2)  { m->minCt++; }
      else if (v>old) { minSortDown(m,p); return; }
      if (minSortUp(m,p) && mmCmpExch(m,0,-1)) { maxSortDown(m,-1); }
   }
   else if (p<0)   //new item is in maxheap
   {  if (m->maxCt < m->N/2) { m->maxCt++; }
      else if (v<old) { maxSortDown(m,p); return; }
      if (maxSortUp(m,p) && m->minCt && mmCmpExch(m,1,0)) { minSortDown(m,1); }
   }
   else //new item is at median
   {  if (m->maxCt && maxSortUp(m,-1)) { maxSortDown(m,-1); }
      if (m->minCt && minSortUp(m, 1)) { minSortDown(m, 1); }
   }
}

//returns median item (or average of 2 when item count is even)
Item MediatorMedian(Mediator* m)
{
   if (m->odd) {
      return m->data[m->heap[0]];
   }
   else
   {
      return (m->data[m->heap[0]] + m->data[m->heap[-1]]) / 2;
   }    
}

