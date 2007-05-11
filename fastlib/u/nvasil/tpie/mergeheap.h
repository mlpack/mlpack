//
// File: mergeheap.h
// Author:  Rakesh Barve <rbarve@cs.duke.edu>
//
// Id: mergeheap.h,v 1.6 1999/11/23 16:49:10 tavi Exp tavi $

// A template useful during merge operations. Basically a heap can be
// maintained on keys; so that the log n comparisons per item involve
// touching an array of keys; not items. The heap is basically the
// heap from CLR except that there is provision to exploit the fact
// that when you are merging you know you will be inserting a new
// element whenever you are doing a delete_min.
//

#ifndef _MERGE_HEAP_H
#define _MERGE_HEAP_H

// Get definitions for working with Unix and Windows
#include <portability.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Macros for left and right.
#define Left(i) 2*i
#define Right(i) 2*i+1
#define Parent(i) i/2

//This is a heap element. Meant to encapsulate the key, along with
//the label run_id indicating the run the key originates from.

template<class Key>
class merge_heap_element {
public:
  Key key;
  unsigned short run_id;
};

/*
TPIE_OS_WIN_ONLY_TEMPLATE_MERGE_HEAP_ELEMENT_COMPILER_FOOLER
*/

//This is the actual heap; there is a constructor, destructor and
//various useful public member functions.

template<class Key>
class merge_heap{

  merge_heap_element<Key> *Heaparray;
  TPIE_OS_SIZE_T Heapsize;
  void Exchange(TPIE_OS_SIZE_T i, TPIE_OS_SIZE_T j) {

    Key tmpkey;
    unsigned short tmpid;

    tmpkey = Heaparray[i].key;
    tmpid = Heaparray[i].run_id;
    
    Heaparray[i].key = Heaparray[j].key;
    Heaparray[i].run_id = Heaparray[j].run_id;
    
    Heaparray[j].key = tmpkey;
    Heaparray[j].run_id = tmpid;
  };

  inline void Heapify(TPIE_OS_SIZE_T i);

public:

  // Constructor
  merge_heap(merge_heap_element<Key> *array_of_elements,
	     TPIE_OS_SIZE_T array_size);

  // Destructor
  ~merge_heap(void) {if (Heaparray) {delete Heaparray; Heaparray = NULL;}};

  // Report size of Heap (number of elements)
  TPIE_OS_SIZE_T sizeofheap(void) {return Heapsize;}; 

  //Delete the current minimum and insert the new item from
  //the same source / run.

  void delete_min_and_insert(Key *nextelement_same_run){
  
    if (nextelement_same_run == NULL) {
      Heaparray[1].key = Heaparray[Heapsize].key;
      Heaparray[1].run_id = Heaparray[Heapsize].run_id;
      Heapsize--;
    } else 
      Heaparray[1].key = *nextelement_same_run;
    this->Heapify(1);
  };

  // Return the minimum key.
  Key get_min_key(void) {return Heaparray[1].key;};

  //Return the run with the minimum key.
  unsigned short get_min_run_id(void) {return Heaparray[1].run_id;};

};



// This is the primary function; note that we have unfolded the 
// recursion.
template<class Key>
inline void merge_heap<Key>::Heapify(TPIE_OS_SIZE_T i) {

  TPIE_OS_SIZE_T l,r, smallest;

  l = Left(i);
  r = Right(i);

  smallest = ((l <= Heapsize) && 
	      (Heaparray[l].key < Heaparray[i].key)) ? l : i;

  smallest = ((r <= Heapsize) && 
	      (Heaparray[r].key < Heaparray[smallest].key))? r : smallest;

  while (smallest != i) {
    this->Exchange(i,smallest);

    i = smallest;
    l = Left(i);
    r = Right(i);
    
    smallest = ((l <= Heapsize) && 
		(Heaparray[l].key < Heaparray[i].key))? l : i;
    
    smallest =  ((r <= Heapsize) && 
		 (Heaparray[r].key < Heaparray[smallest].key))? r : smallest;

  }
}

//Constructor
template<class Key>
merge_heap<Key>::merge_heap(merge_heap_element<Key> *array_of_elements,
			    TPIE_OS_SIZE_T size_of_array) {
  TPIE_OS_SIZE_T i;
  Heapsize = size_of_array;
  Heaparray = array_of_elements;
  for ( i = Heapsize/2; i >= 1; i--) 
    this->Heapify(i);
}

				
template<class Key>
class merge_heap_cmp {

  class merge_heap_element<Key> *Heaparray;
  int (*cmp)(const Key&, const Key&);
  TPIE_OS_SIZE_T Heapsize;

  void Exchange(TPIE_OS_SIZE_T i, TPIE_OS_SIZE_T j){

    Key tmpkey;
    unsigned short tmpid;

    tmpkey = Heaparray[i].key;
    tmpid = Heaparray[i].run_id;
  
    Heaparray[i].key = Heaparray[j].key;
    Heaparray[i].run_id = Heaparray[j].run_id;

    Heaparray[j].key = tmpkey;
    Heaparray[j].run_id = tmpid;
  };

  inline void Heapify(TPIE_OS_SIZE_T i);

public:

  // Constructor
  merge_heap_cmp(class merge_heap_element<Key> *array_of_elements,
		 TPIE_OS_SIZE_T array_size, int (*comp_fun)(const Key&, const Key&));

  // Destructor
  ~merge_heap_cmp(void) {if (Heaparray) {delete Heaparray; Heaparray = NULL;}};

  // Report size of Heap (number of elements)
  TPIE_OS_SIZE_T sizeofheap(void) {return Heapsize;}; 
  
  // Delete the current minimum and insert the new item from
  // the same source / run.

  void delete_min_and_insert(Key *nextelement_same_run) {
  
    if (nextelement_same_run == NULL) {
      Heaparray[1].key = Heaparray[Heapsize].key;
      Heaparray[1].run_id = Heaparray[Heapsize].run_id;
      Heapsize--;
    } else Heaparray[1].key = *nextelement_same_run;
    this->Heapify(1);
  };

  //Return the minimum key.
  Key get_min_key(void) {return Heaparray[1].key;};
  
  //Return the run with the minimum key.
  unsigned short get_min_run_id(void) {return Heaparray[1].run_id;};
};


// This is the primary function; note that we have unfolded the 
// recursion.
template<class Key>
inline void merge_heap_cmp<Key>::Heapify(TPIE_OS_SIZE_T i) {
  TPIE_OS_SIZE_T l,r, smallest;
  
  l = Left(i);
  r = Right(i);

  smallest = ((l <= Heapsize) && (cmp(Heaparray[l].key,Heaparray[i].key)< 0)) ? l : i;

  smallest = ((r <= Heapsize) && 
	      (cmp(Heaparray[r].key,Heaparray[smallest].key)<0))? r : smallest;

  while (smallest != i) {
    this->Exchange(i,smallest);
    
    i = smallest;
    l = Left(i);
    r = Right(i);
    
    smallest = ((l <= Heapsize) && 
		(cmp(Heaparray[l].key,Heaparray[i].key)<0))? l : i;

    smallest =  ((r <= Heapsize) && 
		 (cmp(Heaparray[r].key,Heaparray[smallest].key)<0))? r : smallest;

  }
}

// Constructor
template<class Key>
merge_heap_cmp<Key>::merge_heap_cmp(class merge_heap_element<Key> *array_of_elements,
				    TPIE_OS_SIZE_T size_of_array, 
				    int (*comp_fun)(const Key&, const Key&)) {
  TPIE_OS_SIZE_T i;
  Heapsize = size_of_array;
  Heaparray = array_of_elements;
  cmp = comp_fun;
  
  for ( i = Heapsize/2; i >= 1; i--) 
    this->Heapify(i);
}


#undef Left
#undef Right
#undef Parent
#endif // _MERGE_HEAP_H
