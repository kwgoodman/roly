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
DAMAGE.  */

struct node {
  npy_int64    small; // 1 if the node is in the small heap. 
  npy_int64    idx;   // The node's index in the heap array.
  npy_float64  val;   // The node's value. 
  struct node *next;  // The next node in order of insertion. 
};

struct double_heap {
  npy_int64     n_s;    // The number of elements in the min heap.
  npy_int64     n_l;    // The number of elements in the max heap. 
  struct node **s_heap; // The min heap.
  struct node **l_heap; // The max heap.
  struct node **nodes;    // All the nodes. s_heap and l_heap point
                          // to locations in this array after initialization.
  struct node  *first;    // The node added first to the list of nodes. 
  struct node  *last;     // The last (most recent) node added. 
};


/* 
 * Construct the double heap with the given total number of values. 
 * 
 * Arguments:
 * len -- The total number of values in the double heap. 
 *
 * Return: The double_heap structure, uninitialized. 
 */
struct double_heap dh_create(npy_int64 len) {
  npy_int64 i;
  struct double_heap dh;
  dh.n_l = len / 2;
  dh.n_s = dh.n_l + len % 2;
  dh.nodes = malloc(len * sizeof(struct node*));

  for(i = 0; i < len; ++i) {
    dh.nodes[i] = malloc(sizeof(struct node));
  }
  
  dh.first = dh.nodes[0];
  dh.last  = dh.nodes[dh.n_s + dh.n_l - 1];
  
  dh.s_heap = dh.nodes;
  dh.l_heap = &dh.nodes[dh.n_s];
  return dh;
}


/*
 * Insert initial values into the double heap structure. 
 * 
 * Arguments:
 * dh  -- The double heap structure.
 * idx -- The index of the value running from 0 to len - 1. 
 * val -- The value to insert. 
 */
void dh_insert_init(struct double_heap *dh, npy_int64 idx, npy_float64 val) {
  dh->nodes[idx]->val = val;
}


int _dh_node_comp(const void *lhs, const void *rhs) {
  struct node *l_node = *(struct node**)lhs;
  struct node *r_node = *(struct node**)rhs;

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
void dh_init_median(struct double_heap *dh) {
  npy_int64 i;
  npy_int64 j;
  struct node *tmp;

  // Initialize the next pointers. 
  for(i = 0; i < dh->n_s + dh->n_l - 1; ++i) {
    dh->nodes[i]->next = dh->nodes[i + 1];
  }
  
  qsort(dh->nodes, dh->n_s + dh->n_l, sizeof(struct node*), _dh_node_comp);

  // Reverse the min heap array. 
  for(i = 0; i < dh->n_s / 2; ++i) {
    j = dh->n_s - 1 - i;
    tmp = dh->s_heap[i];
    dh->s_heap[i] = dh->s_heap[j];
    dh->s_heap[j] = tmp;
  }

  // Initialize the nodes' indices. 
  for(i = 0; i < dh->n_s; ++i) {
    dh->s_heap[i]->idx = i;
    dh->s_heap[i]->small = 1;
  }
  for(i = 0; i < dh->n_l; ++i) {
    dh->l_heap[i]->idx = i;
    dh->l_heap[i]->small = 0;
  }
}


/*
 * Return the value of the left child. If there is no left child,
 * return the node's own value.
 */
inline npy_float64 get_l_val(struct node **heap, 
                             npy_int64     len, 
                             npy_int64     idx) {
  npy_int64 l_idx = 2 * idx + 1;
  if(l_idx < len) {
    return heap[l_idx]->val;
  }
  return heap[idx]->val;
}

/*
 * Return the value of the right child. If there is no right child, return 
 * the node's own value. 
 */
inline npy_float64 get_r_val(struct node **heap, 
                             npy_int64     len, 
                             npy_int64     idx) {
  npy_int64 r_idx = 2 * idx + 2;
  if(r_idx < len) {
    return heap[r_idx]->val;
  }
  return heap[idx]->val;
}

/*
 * Return the value of the node's parent. If there is not parent,
 * return the node's own value. 
 */
inline npy_float64 get_p_val(struct node **heap, 
                             npy_int64     len, 
                             npy_int64     idx) {
  if(idx == 0) {
    return heap[idx]->val;
  }
  npy_int64 p_idx = (idx - 1) / 2;
  return heap[p_idx]->val;
}


#define SWAP_NODES(heap, idx1, node1, idx2, node2) \
  node1->idx = idx2; \
  node2->idx = idx1; \
  heap[idx1] = node2; \
  heap[idx2] = node1


/*
 * Swap a node with its left child. 
 */ 
inline npy_int64 swap_left(struct node **heap, npy_int64 c_idx) {
  npy_int64 l_idx = 2 * c_idx + 1;
  struct node *c_node = heap[c_idx];
  struct node *l_node = heap[l_idx];
  SWAP_NODES(heap, c_idx, c_node, l_idx, l_node);
  return l_idx;
}


/*
 * Swap a node with its right child. 
 */
inline npy_int64 swap_right(struct node **heap, npy_int64 c_idx) {
  npy_int64 r_idx = 2 * c_idx + 2;
  struct node *c_node = heap[c_idx];
  struct node *r_node = heap[r_idx];
  SWAP_NODES(heap, c_idx, c_node, r_idx, r_node);
  return r_idx;
}


/*
 * Swap a node with its parent. 
 */ 
inline npy_int64 swap_parent(struct node **heap, npy_int64 c_idx) {
  npy_int64 p_idx = (c_idx - 1) / 2;
  struct node *c_node = heap[c_idx];
  struct node *p_node = heap[p_idx];
  SWAP_NODES(heap, c_idx, c_node, p_idx, p_node);
  return p_idx;
}


/*
 * Move the node at the given index down through the heap to its
 * appropriate position.
 */ 
void move_down_small(struct node **heap, 
                        npy_int64     len, 
                        npy_int64     idx) {
  npy_float64 val = heap[idx]->val;
  npy_float64 l_val, r_val;

  while(1) {
    l_val = get_l_val(heap, len, idx);
    r_val = get_r_val(heap, len, idx);
  
    if(val < l_val || val < r_val) {
      if(l_val > r_val) {
        idx = swap_left(heap, idx);
      } else {
        idx = swap_right(heap, idx);
      }
    } else {
      break; 
    }
  }
}


/*
 * Move the node at the given index up through the heap to its
 * appropriate position. 
 */
void move_up_small(struct node **heap,
                      npy_int64     len,
                      npy_int64     idx) {
  npy_float64 val = heap[idx]->val;
  npy_float64 p_val;
  
  while(1) {
    p_val = get_p_val(heap, len, idx);
    if(val > p_val) {
      idx = swap_parent(heap, idx);
    } else {
      break;
    }
  }
}


/*
 * Move the node at the given index up through the heap to its
 * appropriate position.
 */ 
void move_up_large(struct node **heap, 
                   npy_int64     len, 
                   npy_int64     idx) {
  npy_float64 val = heap[idx]->val;
  npy_float64 l_val, r_val;

  while(1) {
    l_val = get_l_val(heap, len, idx);
    r_val = get_r_val(heap, len, idx);
  
    if(val > l_val || val > r_val) {
      if(l_val < r_val) {
        idx = swap_left(heap, idx);
      } else {
        idx = swap_right(heap, idx);
      }
    } else {
      break; 
    }
  }
}


/*
 * Move the node at the given index down through the heap to its
 * appropriate position. 
 */
void move_down_large(struct node **heap,
                     npy_int64     len,
                     npy_int64     idx) {
  npy_float64 val = heap[idx]->val;
  npy_float64 p_val;
  
  while(1) {
    p_val = get_p_val(heap, len, idx);
    if(val < p_val) {
      idx = swap_parent(heap, idx);
    } else {
      break;
    }
  }
}


/*
 * Rebalance the heaps. This function isn't named very well.
 */
inline void rebalance(struct double_heap *dh) {
  struct node *n = dh->s_heap[0];
  n->small = 0;
  dh->s_heap[0] = dh->l_heap[0];
  dh->s_heap[0]->small = 1;
  dh->l_heap[0] = n;
  move_up_large(dh->l_heap, dh->n_l, 0);
  move_down_small(dh->s_heap, dh->n_s, 0);
}


/*
 * Update the running median with a new value. 
 */
void dh_update(struct double_heap *dh, npy_float64 val) {
  struct node *new_node = dh->first;
  npy_float64 idx = new_node->idx;

  // Replace value of first inserted node, and update first, last.
  new_node->val = val;
  dh->first = dh->first->next;
  dh->last->next = new_node;
  dh->last = new_node;
  
  // In small heap.
  if(new_node->small == 1) {
    move_down_small(dh->s_heap, dh->n_s, idx);
    move_up_small(dh->s_heap, dh->n_s, idx);
  }
  // In max heap. 
  else {
    move_up_large(dh->l_heap, dh->n_l, idx);
    move_down_large(dh->l_heap, dh->n_l, idx);
  }
  
  // Rebalance heaps?
  if(dh->s_heap[0]->val > dh->l_heap[0]->val) {
    rebalance(dh);
  }
}


/*
 * Return the current median value. 
 */
npy_float64 dh_median(struct double_heap *dh) {
  return dh->s_heap[0]->val;
}

/*
 * Print out debugging information. 
 */
void dh_dump(struct double_heap *dh) {
  printf("\n\nFirst: %f\n", dh->first->val);
  printf("Last: %f\n", dh->last->val);
  
  npy_int64 i;
  printf("\n\nSmall heap:\n");
  for(i = 0; i < dh->n_s; ++i) {
    printf("%i, %f\n", dh->s_heap[i]->idx, dh->s_heap[i]->val);
  }
  
  printf("\n\nLarge heap:\n");
  for(i = 0; i < dh->n_l; ++i) {
    printf("%i, %f\n", dh->l_heap[i]->idx, dh->l_heap[i]->val);
  }
}
