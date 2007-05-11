// Copyright (c) 1994 Darren Erik Vengroff
//
// File: pqueue_heap.h
// Author: Darren Erik Vengroff <darrenv@eecs.umich.edu>
// Created: 10/4/94
//
// $Id: pqueue_heap.h,v 1.10 2005/01/14 18:36:24 tavi Exp $
//
// A priority queue class implemented as a binary heap.
//

#ifndef _PQUEUE_HEAP_H
#define _PQUEUE_HEAP_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// The virtual base class that defines what priority queues must do.
template <class T, class P>
class pqueue
{
public:
    // Is it full?
    virtual bool full(void) = 0;
    
    // How many elements?
    virtual unsigned int num_elts(void) = 0;

    // Insert
    virtual bool insert(const T& elt, const P& prio) = 0;

    // Min
    virtual void get_min(T& elt, P& prio) = 0;
    
    // Extract min.
    virtual bool extract_min(T& elt, P& prio) = 0;
};


// Helper functions for navigating through a binary heap.

// The children of an element of the heap.
    
static inline unsigned int lchild(unsigned int index) {
    return 2 * index;
}

static inline unsigned int rchild(unsigned int index) {
    return 2 * index + 1;
}

// The parent of an element.

static inline unsigned int parent(unsigned int index) {
    return index >> 1;
}

template <class T, class P>
struct q_elt {
    T elt;
    P priority;
};


// A base class for priority queues that use heaps.

template <class T, class P>
class pqueue_heap 
{
protected:
    // A pointer to the array of elements and their priorities.
    q_elt<T,P> * elements;

    // The number currently in the queue.
    unsigned int cur_elts;

    // The maximum number the queue can hold.
    unsigned int max_elts;

    // Fix up the heap after a deletion.    
    /* virtual void heapify(unsigned int root) = 0; */

public:
    pqueue_heap(unsigned int size);

    virtual ~pqueue_heap();

    // Is it full?
    bool full(void);

    // How many elements?
    unsigned int num_elts(void);

    // Min
    void get_min(T& elt, P& prio);

};


template <class T, class P>
pqueue_heap<T,P>::pqueue_heap(unsigned int size)
{
    elements = new q_elt<T,P>[max_elts = size];
    cur_elts = 0;
}

template <class T, class P>
pqueue_heap<T,P>::~pqueue_heap() {
    delete [] elements;
    cur_elts = 0;
    max_elts = 0;
    return;
}

template <class T, class P>
bool pqueue_heap<T,P>::full(void) {
    return cur_elts == max_elts;
}

template <class T, class P>
unsigned int pqueue_heap<T,P>::num_elts(void) {
    return cur_elts;
}
    
template <class T, class P>
void pqueue_heap<T,P>::get_min(T& elt, P& prio) {
    elt = elements->elt;
    prio = elements->priority;
}
    

// Comment: (jan) You must not use this version anymore.

// // A priority queue that uses a comparison function for comparing
// // priorities.

// End Comment.


// A priority queue that uses the builtin operator < for comparing
// priorities instead of a comparison function.

template <class T, class P>
class pqueue_heap_op : public pqueue_heap<T,P>
{
private:
    void heapify(unsigned int root);
protected:
    using pqueue_heap<T,P>::cur_elts;
    using pqueue_heap<T,P>::max_elts;
    using pqueue_heap<T,P>::elements;
    
public:
    using pqueue_heap<T,P>::full;
    using pqueue_heap<T,P>::num_elts;
    
    pqueue_heap_op(unsigned int size);
    virtual ~pqueue_heap_op(void) {};

    // Insert
    bool insert(const T& elt, const P& prio);
    // Extract min.
    bool extract_min(T& elt, P& prio);
};

template <class T, class P>
bool pqueue_heap_op<T,P>::extract_min(T& elt, P& prio) {
    if (!cur_elts) {
        return false;
    }
    elt = elements->elt;
    prio = elements->priority;
    elements[0] = elements[--cur_elts];
    heapify(0);

    return true;
}

template <class T, class P>
pqueue_heap_op<T,P>::pqueue_heap_op(unsigned int size) :
        pqueue_heap<T,P>(size)
{
}


template <class T, class P>
bool pqueue_heap_op<T,P>::insert(const T& elt, const P& prio) {
    unsigned int ii;
    
    if (full()) {
        return false;
    }

    for (ii = cur_elts++;
         ii && (elements[parent(ii)].priority > prio);
         ii = parent(ii)) {
        elements[ii] = elements[parent(ii)];
    }
    elements[ii].priority = prio;
    elements[ii].elt = elt;

    return true;
}                                       

template <class T, class P>
void pqueue_heap_op<T,P>::heapify(unsigned int root) {
    unsigned int min_index = root;
    unsigned int lc = lchild(root);
    unsigned int rc = rchild(root);
    
    if ((lc < cur_elts) && (elements[lc].priority <
                            elements[min_index].priority)) {
        min_index = lc;
    }
    if ((rc < cur_elts) && (elements[rc].priority <
                            elements[min_index].priority)) {
        min_index = rc;
    }

    if (min_index != root) {
        q_elt<T,P> tmp_q = elements[min_index];

        elements[min_index] = elements[root];
        elements[root] = tmp_q;

        heapify(min_index);
    }
}   

// A priority queue that uses a comparison object.

template <class T, class P, class CMPR>
class pqueue_heap_obj : public pqueue_heap<T,P>
{
private:
    CMPR *cmp_o;
    void heapify(unsigned int root);
 protected:
    using pqueue_heap<T,P>::cur_elts;
    using pqueue_heap<T,P>::max_elts;    
    using pqueue_heap<T,P>::elements;

 public:
    using pqueue_heap<T,P>::full;
    using pqueue_heap<T,P>::num_elts;

public:
    pqueue_heap_obj(unsigned int size, CMPR *cmp);
    virtual ~pqueue_heap_obj(void) {};

    // Insert
    bool insert(const T& elt, const P& prio);
    // Extract min.
    bool extract_min(T& elt, P& prio);
};

template <class T, class P, class CMPR>
bool pqueue_heap_obj<T,P,CMPR>::extract_min(T& elt, P& prio) {
    if (!cur_elts) {
        return false;
    }
    elt = elements->elt;
    prio = elements->priority;
    elements[0] = elements[--cur_elts];
    heapify(0);

    return true;
}

template <class T, class P, class CMPR>
pqueue_heap_obj<T,P,CMPR>::pqueue_heap_obj(unsigned int size, CMPR *cmp)
        : pqueue_heap<T,P>(size)

{
    cmp_o = cmp;
}


template <class T, class P, class CMPR>
bool pqueue_heap_obj<T,P,CMPR>::insert(const T& elt, const P& prio) {
    unsigned int ii;
    
    if (full()) {
        return false;
    }

    for (ii = cur_elts++;
         ii && (cmp_o->compare(elements[parent(ii)].priority, prio) > 0);
         ii = parent(ii)) {
        elements[ii] = elements[parent(ii)];
    }
    elements[ii].priority = prio;
    elements[ii].elt = elt;

    return true;
}                                       

template <class T, class P, class CMPR>
void pqueue_heap_obj<T,P,CMPR>::heapify(unsigned int root) {
    unsigned int min_index = root;
    unsigned int lc = lchild(root);
    unsigned int rc = rchild(root);
    
    if ((lc < cur_elts) &&
        (cmp_o->compare(elements[lc].priority,
                        elements[min_index].priority) < 0)) {
        min_index = lc;
    }
    if ((rc < cur_elts) &&
        (cmp_o->compare(elements[rc].priority,
                        elements[min_index].priority) < 0)) {
        min_index = rc;
    }

    if (min_index != root) {
        q_elt<T,P> tmp_q = elements[min_index];

        elements[min_index] = elements[root];
        elements[root] = tmp_q;

        heapify(min_index);
    }
}


// Comment: (jan) You must not use this version anymore.

template <class T, class P>
class pqueue_heap_cmp : public pqueue_heap<T,P>
{
  
 private:
  // A pointer to the function used to compare the priorities of
  // elements.
  int (*cmp_f)(const P&, const P&);
  void heapify(unsigned int root);
 protected:
  using pqueue_heap<T,P>::cur_elts;
  using pqueue_heap<T,P>::max_elts;  
  using pqueue_heap<T,P>::elements;
 public:
  using pqueue_heap<T,P>::full;  
  using pqueue_heap<T,P>::num_elts;
                                                                                                                            
 public:
  pqueue_heap_cmp(unsigned int size, int (*cmp)(const P&, const P&));
  virtual ~pqueue_heap_cmp(void) {}                                                                                 
  // Insert
  bool insert(const T& elt, const P& prio);
  // Extract min.
  bool extract_min(T& elt, P& prio);  
};

                                                                                                                            
template <class T, class P>
bool pqueue_heap_cmp<T,P>::extract_min(T& elt, P& prio) 
{
  
  if (!cur_elts) {
      return false;
  }
  
  elt = elements->elt;
  prio = elements->priority;
  elements[0] = elements[--cur_elts];
  heapify(0);                                                                                       
  return true;
  
}

                                                                                                                            
template <class T, class P>
pqueue_heap_cmp<T,P>::pqueue_heap_cmp(unsigned int size,
                                      int (*cmp)(const P&, const P&)) :
  pqueue_heap<T,P>(size) {
  cmp_f = cmp;
}

                                                                                                                            
                                                                                                                            
template <class T, class P>
bool pqueue_heap_cmp<T,P>::insert(const T& elt, const P& prio) 
{
  unsigned int ii;                                                                                                              
  if (full()) {
      return false;
  }
  
                                                                                                                            
  for (ii = cur_elts++;
       ii && (cmp_f(elements[parent(ii)].priority, prio) > 0);
       ii = parent(ii)) 
    {
      
      elements[ii] = elements[parent(ii)];
      
    }
  
  elements[ii].priority = prio;
  
  elements[ii].elt = elt;
  
                                                                                                                            
  return true;
  
}

                                                                                                                            
template <class T, class P>
void pqueue_heap_cmp<T,P>::heapify(unsigned int root) 
{
  
  unsigned int min_index = root;
  
  unsigned int lc = lchild(root);
  
  unsigned int rc = rchild(root);
  
                                                                                                                            
  if ((lc < cur_elts) && (cmp_f(elements[lc].priority,
				elements[min_index].priority) < 0)) 
    {
      
      min_index = lc;
      
    }
  
  if ((rc < cur_elts) && (cmp_f(elements[rc].priority,
				elements[min_index].priority) < 0)) 
    {
      
      min_index = rc;
      
    }
  
                                                                                                                            
  if (min_index != root) 
    {
      
      q_elt<T,P> tmp_q = elements[min_index];
      
                                                                                                                            
      elements[min_index] = elements[root];
      
      elements[root] = tmp_q;
      
                                                                                                                            
      heapify(min_index);
    }
  
}



// // A priority queue that simply uses an array.

// End Comment.

#endif // _PQUEUE_HEAP_H 
