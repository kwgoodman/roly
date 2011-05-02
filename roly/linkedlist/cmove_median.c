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

/* 
 * Initialize (given an array values):

   n = 255;
   struct mm_list mm = mm_new(n);
   for(i = 0; i < n; ++i) {
     mm_insert_init(&mm, values(i));
   }
   mm_init_median(&mm);

 * Update:

   mm_update(&mm, value);

 * Get median value:
 
   median = mm_get_median(&mm);
  
 */

struct mm_node {
  npy_float64 val;              // The node's value. 
  struct mm_node *next_idx;     // The next node in order of index.
  struct mm_node *larger_node;  // The next larger node sorted by value.
  struct mm_node *smaller_node; // The next smaller node sorted by value.
};

struct mm_node *new_node(npy_float64 val) {
  struct mm_node *n = (struct mm_node*)malloc(sizeof(struct mm_node));
  n->val = val;
  n->next_idx = 0;
  n->larger_node = 0;
  n->smaller_node = 0;
  return n;
}

struct mm_list {
  npy_int64 len;            // The length of the list.
  struct mm_node *head;     // The head node by index.
  struct mm_node *tail;     // The tail node by index.
  struct mm_node *min_node; // The smallest node by value.
  struct mm_node *med_node; // The middle element by value.
};

struct mm_list mm_new(npy_int64 len) {
  struct mm_list mm;
  mm.len = len;
  mm.head = 0;
  mm.tail = 0;
  mm.min_node = 0;
  mm.med_node = 0;
  return mm;
}

void mm_insert_after(struct mm_node *n, struct mm_node *n_new) {
  n_new->smaller_node = n;
  n_new->larger_node = n->larger_node;
  n->larger_node = n_new;
  if(n_new->larger_node != 0) {
    n_new->larger_node->smaller_node = n_new;
  }
}


void mm_insert_init(struct mm_list *mm, npy_float64 val) {
  struct mm_node *n_new = new_node(val);

  // If this is the first node. 
  if(mm->tail == 0) {
    mm->head = mm->tail = mm->min_node = n_new;
    return;
  }
  
  // Insert the node at the tail by index.
  mm->tail->next_idx = n_new;
  mm->tail = n_new;
  struct mm_node *n = mm->min_node;
  
  // New smallest node?
  if(n_new->val < n->val) {
    n_new->larger_node = n;
    n->smaller_node = n_new;
    mm->min_node = n_new;
    return;
  }

  // Find node to insert after. 
  while(n->larger_node != 0 && n_new->val > n->larger_node->val) {
    n = n->larger_node;
  }
  
  // Insert after this node. 
  mm_insert_after(n, n_new);
}

void mm_init_median(struct mm_list *mm) {
  npy_int64 i;
  mm->med_node = mm->min_node;
  for(i = 0; i < mm->len/2; ++i) {
    mm->med_node = mm->med_node->larger_node;
  }
}

npy_float64 mm_get_median(struct mm_list *mm) {
  return mm->med_node->val;
}

inline void mm_swap_nodes(struct mm_list *mm, 
                          struct mm_node *ln, 
                          struct mm_node *rn) {
  // rn is the right side node, ln is the left side node. 
  ln->larger_node = rn->larger_node;
  rn->smaller_node = ln->smaller_node;
  ln->smaller_node = rn;
  rn->larger_node = ln;
  
  if(ln->larger_node != 0) {
    ln->larger_node->smaller_node = ln;
  }
  if(rn->smaller_node != 0) {
    rn->smaller_node->larger_node = rn;
  } else {
    mm->min_node = rn;
  }
  
  // Update the median node. 
  if(ln == mm->med_node) {
    mm->med_node = rn;
  } else if(rn == mm->med_node) {
    mm->med_node = ln;
  }
}

void mm_update(struct mm_list *mm, npy_float64 val) {
  // Remove the head node and move it to the tail.
  struct mm_node *n_new = mm->head;
  mm->head = n_new->next_idx;
  n_new->next_idx = 0;
  mm->tail->next_idx = n_new;
  mm->tail = n_new;

  // Update the value.
  n_new->val = val;
  
  // Move right.
  while(n_new->larger_node != 0 && n_new->val > n_new->larger_node->val) {
    mm_swap_nodes(mm, n_new, n_new->larger_node);
  }
  
  // Move left.
  while(n_new->smaller_node != 0 && n_new->val < n_new->smaller_node->val) {
    mm_swap_nodes(mm, n_new->smaller_node, n_new);
  }
}

void mm_free(struct mm_list *mm) {
  struct mm_node *n = mm->head;
  struct mm_node *tmp;
  while(n != 0) {
    tmp = n->next_idx;
    free(n);
    n = tmp;
  }
}
