/*
Copyright (c) 2011 J. David Lee. All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.  
*/

#define NUM_CHILDREN 2
#define min(a, b) (((a) < (b)) ? (a) : (b))

struct mm_node {
  npy_int64    small; // 1 if the node is in the small heap. 
  npy_int64    idx;   // The node's index in the heap array.
  npy_float64  val;   // The node's value. 
  struct mm_node *next;  // The next node in order of insertion. 
};


struct mm_handle {
  npy_int64        n_tot;  // The total number of elements in nodes below.
  npy_int64        n_s;    // The number of elements in the min heap.
  npy_int64        n_l;    // The number of elements in the max heap. 
  struct mm_node **s_heap; // The min heap.
  struct mm_node **l_heap; // The max heap.
  struct mm_node **nodes;  // All the nodes. s_heap and l_heap point
                           // to locations in this array after initialization.
  struct mm_node  *first;  // The node added first to the list of nodes. 
  struct mm_node  *last;   // The last (most recent) node added. 
};


/* 
 * Construct the double heap with the given total number of values. 
 * 
 * Arguments:
 * len -- The total number of values in the double heap. 
 *
 * Return: The mm_handle structure, uninitialized. 
 */
struct mm_handle *mm_new(npy_int64 len) {
  npy_int64 i;
  struct mm_handle *mm = malloc(sizeof(struct mm_handle));
  mm->n_tot = len;
  mm->n_l = len / 2;
  mm->n_s = mm->n_l + len % 2;
  mm->nodes = malloc(len * sizeof(struct mm_node*));

  for(i = 0; i < len; ++i) {
    mm->nodes[i] = malloc(sizeof(struct mm_node));
  }
  
  mm->first = mm->nodes[0];
  mm->last  = mm->nodes[mm->n_tot - 1];
  
  mm->s_heap = mm->nodes;
  mm->l_heap = &mm->nodes[mm->n_s];
  return mm;
}


/*
 * Insert initial values into the double heap structure. 
 * 
 * Arguments:
 * mm  -- The double heap structure.
 * idx -- The index of the value running from 0 to len - 1. 
 * val -- The value to insert. 
 */
inline void mm_insert_init(struct mm_handle *mm, 
                           npy_int64 idx, 
                           npy_float64 val) {
  mm->nodes[idx]->val = val;
}


inline int _node_comp(const void *lhs, const void *rhs) {
  struct mm_node *l_node = *(struct mm_node**)lhs;
  struct mm_node *r_node = *(struct mm_node**)rhs;

  if(l_node->val < r_node->val) {
    return -1;
  }
  if(l_node->val > r_node->val) {
    return 1;
  }
  return 0;
}


/*
 * Initialize the double heap structure to find the median. 
 */
void mm_init_median(struct mm_handle *mm) {
  npy_int64 i;
  npy_int64 j;
  struct mm_node *tmp;

  // Initialize the next pointers. 
  for(i = 0; i < mm->n_s + mm->n_l - 1; ++i) {
    mm->nodes[i]->next = mm->nodes[i + 1];
  }
  
  qsort(mm->nodes, mm->n_tot, sizeof(struct mm_node*), _node_comp);

  // Reverse the min heap array. 
  for(i = 0; i < mm->n_s / 2; ++i) {
    j = mm->n_s - 1 - i;
    tmp = mm->s_heap[i];
    mm->s_heap[i] = mm->s_heap[j];
    mm->s_heap[j] = tmp;
  }

  // Initialize the nodes' indices. 
  for(i = 0; i < mm->n_s; ++i) {
    mm->s_heap[i]->idx = i;
    mm->s_heap[i]->small = 1;
  }
  for(i = 0; i < mm->n_l; ++i) {
    mm->l_heap[i]->idx = i;
    mm->l_heap[i]->small = 0;
  }
}


/*
 * Return the smallest child node of the given node, if it doesn't exist, 
 * return the node itself. 
 */
inline struct mm_node *get_smallest_child(struct mm_node **heap,
                                          npy_int64        len,
                                          struct mm_node  *node) {
  npy_int64 i     = NUM_CHILDREN * node->idx + 1;
  npy_int64 i_end = min(i + NUM_CHILDREN, len);
  struct mm_node *child;
  
  for(; i < i_end; ++i) {
    child = heap[i];
    if(child->val < node->val) {
      node = child;
    }
  }
  return node;
}


/*
 * Return the largest child node of the given node, if it doesn't exist, 
 * return the node itself. 
 */
inline struct mm_node *get_largest_child(struct mm_node **heap,
                                         npy_int64        len,
                                         struct mm_node  *node) {
  npy_int64 i     = NUM_CHILDREN * node->idx + 1;
  npy_int64 i_end = min(i + NUM_CHILDREN, len);
  struct mm_node *child;

  for(; i < i_end; ++i) {
    child = heap[i];
    if(child->val > node->val) {
      node = child;
    }
  }
  return node;
}


/*
 * Return the node's parent.
 */
inline struct mm_node *get_parent(struct mm_node **heap,
                                  struct mm_node  *node) {
  npy_int64 p_idx = (node->idx - 1) / NUM_CHILDREN;
  return heap[p_idx];
}


/* 
 * Swap two nodes. 
 */ 
inline void swap_nodes(struct mm_node **heap, 
                       struct mm_node  *node1, 
                       struct mm_node  *node2) {
  npy_int64 idx1 = node1->idx;
  npy_int64 idx2 = node2->idx;
  heap[idx1] = node2;
  heap[idx2] = node1;
  node1->idx = idx2;
  node2->idx = idx1;
}


/*
 * Move the given node up through the heap to the appropriate position. 
 */
inline void move_up_small(struct mm_node **heap,
                          npy_int64        len,
                          struct mm_node  *node) {
  struct mm_node *parent = get_parent(heap, node);
  while(node->idx != 0 && node->val > parent->val) {
    swap_nodes(heap, node, parent);
    parent = get_parent(heap, node);
  }
}

/*
 * Move the given node down through the heap to the appropriate position. 
 */ 
inline void move_down_small(struct mm_node **heap,
                            npy_int64        len,
                            struct mm_node  *node) {
  struct mm_node *child = get_largest_child(heap, len, node);
  while(node->val < child->val) {
    swap_nodes(heap, node, child);
    child = get_largest_child(heap, len, node);
  }
}


/*
 * Move the given node down through the heap to the appropriate
 * position.
 */ 
inline void move_down_large(struct mm_node **heap,
                            npy_int64        len,
                            struct mm_node  *node) {
  struct mm_node *parent = get_parent(heap, node);
  while(node->idx != 0 && node->val < parent->val) {
    swap_nodes(heap, node, parent);
    parent = get_parent(heap, node);
  }
}


/*
 * Move the given node down through the heap to the appropriate position. 
 */ 
inline void move_up_large(struct mm_node **heap,
                          npy_int64        len,
                          struct mm_node  *node) {
  struct mm_node *child = get_smallest_child(heap, len, node);
  while(node->val > child->val) {
    swap_nodes(heap, node, child);
    child = get_smallest_child(heap, len, node);
  }
}


/*
 * Rebalance the heaps.
 */
inline void swap_heap_heads(struct mm_handle *mm) {
  struct mm_node *s_node = mm->s_heap[0];
  struct mm_node *l_node = mm->l_heap[0];
  l_node->small = 1;
  s_node->small = 0;
  mm->s_heap[0] = l_node;
  mm->l_heap[0] = s_node;
  move_up_large(mm->l_heap, mm->n_l, s_node);
  move_down_small(mm->s_heap, mm->n_s, l_node);
}


/*
 * Update the running median with a new value. 
 */
inline void mm_update(struct mm_handle *mm, npy_float64 val) {
  struct mm_node *new_node = mm->first;

  // Replace value of first inserted node, and update first, last.
  new_node->val = val;
  mm->first = mm->first->next;
  mm->last->next = new_node;
  mm->last = new_node;

  // In small heap.
  if(new_node->small == 1) {
    if(new_node->idx != 0 && 
       val > get_parent(mm->s_heap, new_node)->val) {
      move_up_small(mm->s_heap, mm->n_s, new_node);
    } else {
      move_down_small(mm->s_heap, mm->n_s, new_node);
    }
    if(val > mm->l_heap[0]->val) {
      swap_heap_heads(mm);
    }
  }

  // In large heap. 
  else {
    if(new_node->idx != 0 &&
       val < get_parent(mm->l_heap, new_node)->val) {
      move_down_large(mm->l_heap, mm->n_l, new_node);
    } else {
      move_up_large(mm->l_heap, mm->n_l, new_node);
    }
    if(val < mm->s_heap[0]->val) {
      swap_heap_heads(mm);
    }
  }
}


/*
 * Return the current median value. 
 */
inline npy_float64 mm_get_median(struct mm_handle *mm) {
  return mm->s_heap[0]->val;
}


/*
void mm_dump(struct mm_handle *mm) {
  printf("\n\nFirst: %f\n", mm->first->val);
  printf("Last: %f\n", mm->last->val);
  
  npy_int64 i;
  printf("\n\nSmall heap:\n");
  for(i = 0; i < mm->n_s; ++i) {
    printf("%l %f\n", mm->s_heap[i]->idx, mm->s_heap[i]->val);
  }
  
  printf("\n\nLarge heap:\n");
  for(i = 0; i < mm->n_l; ++i) {
    printf("%l %f\n", mm->l_heap[i]->idx, mm->l_heap[i]->val);
  }
}
*/


/*
 * Free memory allocated in mm_new.
 */
inline void mm_free(struct mm_handle *mm) {
  npy_int64 i;
  for(i = 0; i < mm->n_tot; ++i) {
    free(mm->nodes[i]);
  }
  free(mm->nodes);
  free(mm);
}
