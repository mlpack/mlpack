//
// File: mergeheap_dh.h
// 
// $Id: mergeheap_dh.h,v 1.9 2005/08/24 19:32:38 adanner Exp $	

// This file contains several merge heap templates. 
// Originally written by Rakesh Barve.  

// The heap is basically the heap from CLR except that there is
// provision to exploit the fact that when you are merging you know
// you will be inserting a new element whenever you are doing a
// delete_min.

// Modified by David Hutchinson 2000 03 02

//     - main purpose of the mods is to allow the merge heap to be
//     part of a sort management object. The sort management object
//     contains several procedures and data structures needed for
//     sorting, but the precise versions of these procedures and data
//     structures depend on the sorting approach used (this permits
//     parameterization of the sorting procedure via the sort
//     management object, and avoids having multiple versions of large
//     pieces of code that are highly redundant and difficult to
//     maintain).

//     - move initialization from constructor to an explicit
//     "initialize" member function

//     - add a "comparison object" version of the merge heap
//     object. This allows a comparison object with a "compare" member
//     function to be specified for comparing keys. "Comparison
//     operator" and "comparison function" versions previously
//     existed.

//     - add a set of three (comparison object, operator and function)
//     versions of the merge heap that maintain pointers to the
//     current records at the head of the streams being merged. The
//     previous versions kept the entire corresponding record in the heap.

#ifndef _MERGE_HEAP_DH_H
#define _MERGE_HEAP_DH_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// Macros for left and right.
#define Left(i)   2*(i)
#define Right(i)  2*(i)+1
#define Parent(i) (i)/2

// This is a heap element. Encapsulates the key, along with
// the label run_id indicating the run the key originates from.

template<class KEY>
class heap_element {
public:
    heap_element(){};
    KEY            key;
    TPIE_OS_SIZE_T run_id;
};

// This is a record pointer element. Encapsulates the record pointer,
// along with the label run_id indicating the run the record
// originates from.

template<class REC>
class heap_ptr {
public:
    heap_ptr(){};
    ~heap_ptr(){};
    REC            *recptr;
    TPIE_OS_SIZE_T run_id;
};

// ********************************************************************
// * A record pointer heap base class - also serves as the full       *
// * implementation for objects with a < comparison operator         *
// ********************************************************************

template<class REC>
class merge_heap_pdh_op{

protected:

  heap_ptr<REC> *Heaparray;
  TPIE_OS_SIZE_T  Heapsize;
  TPIE_OS_SIZE_T  maxHeapsize;
  
  inline void Exchange(TPIE_OS_SIZE_T i, TPIE_OS_SIZE_T j); 
  
  //These functions will typically be overridden by subclasses
  inline TPIE_OS_SIZE_T get_smallest(TPIE_OS_SIZE_T i);
  inline void Heapify(TPIE_OS_SIZE_T i);

public:

  // Constructor/Destructor 
  merge_heap_pdh_op() { Heaparray=NULL; };
  ~merge_heap_pdh_op() { 
     //Cleanup if someone forgot de-allocate
     //(abd) This seems to cause double free errors, but I don't know why
     //This was just a safeguard anyways, turn off for now
     //if(Heaparray != NULL){delete [] Heaparray;}
  }

  // Report size of Heap (number of elements)
  TPIE_OS_SIZE_T sizeofheap(void) {return Heapsize;}; 

  // Return the run with the minimum key.
  inline TPIE_OS_SIZE_T get_min_run_id(void) {return Heaparray[1].run_id;};

  void allocate   (TPIE_OS_SIZE_T size);
  void insert     (REC *ptr, TPIE_OS_SIZE_T run_id);
  void deallocate (void);
  
  // heapify's an initial array of elements
  // typically overridden in sub class. 
  void initialize (void);
  
  // Delete the current minimum and insert the new item from the same
  // source / run.
  inline void delete_min_and_insert(REC *nextelement_same_run);

  // Return main memory space usage per item
  inline TPIE_OS_SIZE_T space_per_item(void) { return sizeof(heap_ptr<REC>); }
  
  // Return fixed main memory space overhead, regardless of item count
  inline TPIE_OS_SIZE_T space_overhead(void) { 
    // One extra array item is defined to make heap indexing easier
    return sizeof(heap_ptr<REC>)+MM_manager.space_overhead();
  }
  
};

template<class REC>
inline void merge_heap_pdh_op<REC>::Exchange(TPIE_OS_SIZE_T i,
                                               TPIE_OS_SIZE_T j)
{
  REC* tmpptr;
  TPIE_OS_SIZE_T tmpid;
	tmpptr = Heaparray[i].recptr;
	tmpid = Heaparray[i].run_id;   
	Heaparray[i].recptr = Heaparray[j].recptr;
	Heaparray[i].run_id = Heaparray[j].run_id;
	Heaparray[j].recptr = tmpptr;
	Heaparray[j].run_id = tmpid;
}

//Returns the index of the smallest element out of
//i, the left child of i, and the right child of i
template<class REC>
inline TPIE_OS_SIZE_T merge_heap_pdh_op<REC>::get_smallest(
                                                      TPIE_OS_SIZE_T i)
{
  TPIE_OS_SIZE_T l,r, smallest;
 
  l = Left(i);
  r = Right(i);

  smallest = ((l <= Heapsize) && 
      (*Heaparray[l].recptr < *Heaparray[i].recptr)) ? l : i;

  smallest = ((r <= Heapsize) && 
      (*Heaparray[r].recptr < *Heaparray[smallest].recptr))? r : smallest;

  return smallest;
}

// This is the primary function; note that we have unfolded the 
// recursion.
template<class REC>
inline void merge_heap_pdh_op<REC>::Heapify(TPIE_OS_SIZE_T i) {

    TPIE_OS_SIZE_T smallest = get_smallest(i);
    
    while (smallest != i) {
      this->Exchange(i,smallest);
      i = smallest;
      smallest = get_smallest(i);
    }
}

template<class REC>
inline void merge_heap_pdh_op<REC>::delete_min_and_insert
                                     (REC *nextelement_same_run)
{ 
  if (nextelement_same_run == NULL) {
    Heaparray[1].recptr = Heaparray[Heapsize].recptr;
    Heaparray[1].run_id = Heaparray[Heapsize].run_id;
    Heapsize--;
  } else { 
    Heaparray[1].recptr = nextelement_same_run;
  }
  Heapify(1);
}

// Allocate space for the heap
template<class REC>
inline void merge_heap_pdh_op<REC>::allocate ( TPIE_OS_SIZE_T size ) {
    Heaparray = new heap_ptr<REC> [size+1];
    Heapsize  = 0;
    maxHeapsize = size;
}

// Copy an (initial) element into the heap array
template<class REC>
inline void merge_heap_pdh_op<REC>::insert (REC *ptr, TPIE_OS_SIZE_T run_id)
{
    Heaparray[Heapsize+1].recptr    = ptr;
    Heaparray[Heapsize+1].run_id = run_id;
    Heapsize++;
}

// Deallocate the space used by the heap
template<class REC>
inline void merge_heap_pdh_op<REC>::deallocate () {
  if (Heaparray){
    delete [] Heaparray; 
    Heaparray=NULL;
  }
  Heapsize    = 0;
  maxHeapsize = 0;
}

template<class REC>
void merge_heap_pdh_op<REC>::initialize () {
  for ( TPIE_OS_SIZE_T i = Heapsize/2; i >= 1; i--){ Heapify(i); }
}

// ********************************************************************
// * A record pointer heap that uses a comparison object              *
// ********************************************************************

template<class REC, class CMPR>
class merge_heap_pdh_obj: public merge_heap_pdh_op<REC>{

protected: 
  
  using merge_heap_pdh_op<REC>::Heapsize;
  using merge_heap_pdh_op<REC>::Heaparray;
  using merge_heap_pdh_op<REC>::maxHeapsize;
  CMPR* cmp;
  
  inline TPIE_OS_SIZE_T get_smallest(TPIE_OS_SIZE_T i);
  inline void Heapify(TPIE_OS_SIZE_T i);

public:
  using merge_heap_pdh_op<REC>::sizeofheap;
  
  // Constructor initializes a pointer to the user's comparison object
  // The object may contain dynamic data although the 'compare' method is const
  // and therefore inline'able.
  merge_heap_pdh_obj ( CMPR *cmptr ) : cmp(cmptr) {};
  ~merge_heap_pdh_obj(){};
  
  void initialize (void);
  
  // Delete the current minimum and insert the new item from the same
  // source / run.
  inline void delete_min_and_insert(REC *nextelement_same_run);
};

//Returns the index of the smallest element out of
//i, the left child of i, and the right child of i
template<class REC, class CMPR>
inline TPIE_OS_SIZE_T merge_heap_pdh_obj<REC,CMPR>::get_smallest(
                                                      TPIE_OS_SIZE_T i)
{
  TPIE_OS_SIZE_T l,r, smallest;
 
  l = Left(i);
  r = Right(i);

  smallest = ((l <= Heapsize) && 
    (cmp->compare(*Heaparray[l].recptr,*Heaparray[i].recptr)< 0)) ? l : i;

  smallest = ((r <= Heapsize) && 
    (cmp->compare(*Heaparray[r].recptr,*Heaparray[smallest].recptr)<0))?
    r : smallest;

  return smallest;
}

template<class REC, class CMPR>
inline void merge_heap_pdh_obj<REC, CMPR>::Heapify(TPIE_OS_SIZE_T i) {
    TPIE_OS_SIZE_T smallest = get_smallest(i);
    while (smallest != i) {
      this->Exchange(i,smallest);
      i = smallest;
      smallest = get_smallest(i);
    }
}

template<class REC, class CMPR>
inline void merge_heap_pdh_obj<REC, CMPR>::delete_min_and_insert
                                     (REC *nextelement_same_run)
{ 
  if (nextelement_same_run == NULL) {
    Heaparray[1].recptr = Heaparray[Heapsize].recptr;
    Heaparray[1].run_id = Heaparray[Heapsize].run_id;
    Heapsize--;
  } else { 
    Heaparray[1].recptr = nextelement_same_run;
  }
  Heapify(1);
}


template<class REC, class CMPR>
void merge_heap_pdh_obj<REC, CMPR>::initialize () {
  for ( TPIE_OS_SIZE_T i = Heapsize/2; i >= 1; i--){ Heapify(i); }
}

// ********************************************************************
// * A merge heap object base class - also serves as the full         *
// * implementation for objects with a < comparison operator          *
// ********************************************************************

template<class REC>
class merge_heap_dh_op{

protected:

  heap_element<REC> *Heaparray;
  TPIE_OS_SIZE_T  Heapsize;
  TPIE_OS_SIZE_T  maxHeapsize;
  
  inline void Exchange(TPIE_OS_SIZE_T i, TPIE_OS_SIZE_T j); 
  inline void Heapify(TPIE_OS_SIZE_T i);
  //This function will typically be overridden by subclasses
  inline TPIE_OS_SIZE_T get_smallest(TPIE_OS_SIZE_T i);

public:

  // Constructor/Destructor 
  merge_heap_dh_op() { Heaparray=NULL; };
  ~merge_heap_dh_op() { 
     //Cleanup if someone forgot de-allocate
     //(abd) This seems to cause double free errors, but I don't know why
     //This was just a safeguard anyways, turn off for now
     //if(Heaparray != NULL){delete [] Heaparray;}
  }

  // Report size of Heap (number of elements)
  TPIE_OS_SIZE_T sizeofheap(void) {return Heapsize;}; 

  // Return the run with the minimum key.
  inline TPIE_OS_SIZE_T get_min_run_id(void) {return Heaparray[1].run_id;};

  void allocate   (TPIE_OS_SIZE_T size);
  void insert     (REC *ptr, TPIE_OS_SIZE_T run_id);
  void deallocate (void);
  
  // heapify's an initial array of elements
  void initialize (void);
  
  // Delete the current minimum and insert the new item from the same
  // source / run.
  inline void delete_min_and_insert(REC *nextelement_same_run);

  // Return main memory space usage per item
  inline TPIE_OS_SIZE_T space_per_item(void) {
    return sizeof(heap_element<REC>);
  }
  
  // Return fixed main memory space overhead, regardless of item count
  inline TPIE_OS_SIZE_T space_overhead(void) { 
    // One extra array item is defined to make heap indexing easier
    return sizeof(heap_element<REC>)+MM_manager.space_overhead();
  }
  
};

template<class REC>
inline void merge_heap_dh_op<REC>::Exchange(TPIE_OS_SIZE_T i,
                                               TPIE_OS_SIZE_T j)
{
  REC tmpkey;
  TPIE_OS_SIZE_T tmpid;
	tmpkey = Heaparray[i].key;
	tmpid = Heaparray[i].run_id;   
	Heaparray[i].key = Heaparray[j].key;
	Heaparray[i].run_id = Heaparray[j].run_id;
	Heaparray[j].key = tmpkey;
	Heaparray[j].run_id = tmpid;
}

//Returns the index of the smallest element out of
//i, the left child of i, and the right child of i
template<class REC>
inline TPIE_OS_SIZE_T merge_heap_dh_op<REC>::get_smallest(
                                                      TPIE_OS_SIZE_T i)
{
  TPIE_OS_SIZE_T l,r, smallest;
 
  l = Left(i);
  r = Right(i);

  smallest = ((l <= Heapsize) && 
      (Heaparray[l].key < Heaparray[i].key)) ? l : i;

  smallest = ((r <= Heapsize) && 
      (Heaparray[r].key < Heaparray[smallest].key))? r : smallest;

  return smallest;
}

// This is the primary function; note that we have unfolded the 
// recursion.
template<class REC>
inline void merge_heap_dh_op<REC>::Heapify(TPIE_OS_SIZE_T i) {

    TPIE_OS_SIZE_T smallest = get_smallest(i);
    
    while (smallest != i) {
      this->Exchange(i,smallest);
      i = smallest;
      smallest = get_smallest(i);
    }
}

template<class REC>
inline void merge_heap_dh_op<REC>::delete_min_and_insert
                                     (REC *nextelement_same_run)
{ 
  if (nextelement_same_run == NULL) {
    Heaparray[1].key = Heaparray[Heapsize].key;
    Heaparray[1].run_id = Heaparray[Heapsize].run_id;
    Heapsize--;
  } else { 
    Heaparray[1].key = *nextelement_same_run;
  }
  Heapify(1);
}

// Allocate space for the heap
template<class REC>
inline void merge_heap_dh_op<REC>::allocate ( TPIE_OS_SIZE_T size ) {
    Heaparray = new heap_element<REC> [size+1];
    Heapsize  = 0;
    maxHeapsize = size;
}

// Copy an (initial) element into the heap array
template<class REC>
inline void merge_heap_dh_op<REC>::insert (REC *ptr, TPIE_OS_SIZE_T run_id)
{
    Heaparray[Heapsize+1].key    = *ptr;
    Heaparray[Heapsize+1].run_id = run_id;
    Heapsize++;
}

// Deallocate the space used by the heap
template<class REC>
inline void merge_heap_dh_op<REC>::deallocate () {
  if (Heaparray){
    delete [] Heaparray; 
    Heaparray=NULL;
  }
  Heapsize    = 0;
  maxHeapsize = 0;
};

template<class REC>
void merge_heap_dh_op<REC>::initialize () {
  for ( TPIE_OS_SIZE_T i = Heapsize/2; i >= 1; i--){ Heapify(i); }
}


// ********************************************************************
// * A merge heap that uses a comparison object                       *
// ********************************************************************

template<class REC, class CMPR>
class merge_heap_dh_obj: public merge_heap_dh_op<REC>{

protected:   
  using merge_heap_dh_op<REC>::Heapsize;
  using merge_heap_dh_op<REC>::Heaparray;
  using merge_heap_dh_op<REC>::maxHeapsize;
  CMPR* cmp;
  
  inline TPIE_OS_SIZE_T get_smallest(TPIE_OS_SIZE_T i);

  inline void Heapify(TPIE_OS_SIZE_T i);
public:
  using merge_heap_dh_op<REC>::sizeofheap;
  
  // Constructor initializes a pointer to the user's comparison object
  // The object may contain dynamic data although the 'compare' method is const
  // and therefore inline'able.
  merge_heap_dh_obj ( CMPR *cmptr ) : cmp(cmptr) {};
  ~merge_heap_dh_obj(){};
  
  // heapify's an initial array of elements
  void initialize (void);
  
  // Delete the current minimum and insert the new item from the same
  // source / run.
  inline void delete_min_and_insert(REC *nextelement_same_run);
};

//Returns the index of the smallest element out of
//i, the left child of i, and the right child of i
template<class REC, class CMPR>
inline TPIE_OS_SIZE_T merge_heap_dh_obj<REC,CMPR>::get_smallest(
                                                      TPIE_OS_SIZE_T i)
{
  TPIE_OS_SIZE_T l,r, smallest;
 
  l = Left(i);
  r = Right(i);

  smallest = ((l <= Heapsize) &&
              (cmp->compare(Heaparray[l].key,Heaparray[i].key)< 0)) ? l : i;

  smallest = ((r <= Heapsize) && 
		          (cmp->compare(Heaparray[r].key,Heaparray[smallest].key)<0))?
              r : smallest;

  return smallest;
}

template<class REC, class CMPR>
inline void merge_heap_dh_obj<REC, CMPR>::Heapify(TPIE_OS_SIZE_T i) {
    TPIE_OS_SIZE_T smallest = get_smallest(i);
    while (smallest != i) {
      this->Exchange(i,smallest);
      i = smallest;
      smallest = get_smallest(i);
    }
}

template<class REC, class CMPR>
inline void merge_heap_dh_obj<REC, CMPR>::delete_min_and_insert
                                     (REC *nextelement_same_run)
{ 
  if (nextelement_same_run == NULL) {
    Heaparray[1].key = Heaparray[Heapsize].key;
    Heaparray[1].run_id = Heaparray[Heapsize].run_id;
    Heapsize--;
  } else { 
    Heaparray[1].key = *nextelement_same_run;
  }
  Heapify(1);
}


template<class REC, class CMPR>
void merge_heap_dh_obj<REC, CMPR>::initialize () {
  for ( TPIE_OS_SIZE_T i = Heapsize/2; i >= 1; i--){ Heapify(i); }
}

// ********************************************************************
// * A merge heap key-object base class                               *
// * Also serves as a full impelementation of a                       *
// * key-merge heap that uses a comparison operator <                 *
// ********************************************************************

// The merge_heap_dh_kop object maintains only the keys in its heap,
// and uses the member function "copy" of the user-provided class CMPR
// to copy these keys from each record.

template<class REC, class KEY, class CMPR>
class merge_heap_dh_kop{

protected:

  heap_element<KEY> *Heaparray;
  TPIE_OS_SIZE_T  Heapsize;
  TPIE_OS_SIZE_T  maxHeapsize;
  inline void Exchange(TPIE_OS_SIZE_T i, TPIE_OS_SIZE_T j);
  inline void Heapify(TPIE_OS_SIZE_T i);
  //This function will typically be overridden by subclasses
  inline TPIE_OS_SIZE_T get_smallest(TPIE_OS_SIZE_T i);
  CMPR *UsrObject;
  
public:

  // Constructor/Destructor 
  merge_heap_dh_kop( CMPR* cmpptr) : UsrObject(cmpptr), Heaparray(NULL) {};
  ~merge_heap_dh_kop() { 
     //Cleanup if someone forgot de-allocate
     //(abd) This seems to cause double free errors, but I don't know why
     //This was just a safeguard anyways, turn off for now
     //if(Heaparray != NULL){delete [] Heaparray;}
  }

  // Report size of Heap (number of elements)
  TPIE_OS_SIZE_T sizeofheap(void) {return Heapsize;}; 

  // Return the run with the minimum key.
  inline TPIE_OS_SIZE_T get_min_run_id(void) {return Heaparray[1].run_id;};

  void allocate   (TPIE_OS_SIZE_T size);
  void insert     (REC *ptr, TPIE_OS_SIZE_T run_id);
  void deallocate ();
  
  // Delete the current minimum and insert the new item from the same
  // source / run.
  inline void delete_min_and_insert(REC *nextelement_same_run);
  
  // Return main memory space usage per item
  inline TPIE_OS_SIZE_T space_per_item(void) {
    return sizeof(heap_element<REC>);
  }
  
  // Return fixed main memory space overhead, regardless of item count
  inline TPIE_OS_SIZE_T space_overhead(void) { 
    // One extra array item is defined to make heap indexing easier
    return sizeof(heap_ptr<REC>)+MM_manager.space_overhead();
  }
  
  // heapify's an initial array of elements
  void initialize (void);
  
};

template<class REC, class KEY, class CMPR>
inline void merge_heap_dh_kop<REC,KEY,CMPR>::Exchange(TPIE_OS_SIZE_T i,
                                                      TPIE_OS_SIZE_T j)
{
  KEY tmpkey;
  TPIE_OS_SIZE_T tmpid;
	tmpkey = Heaparray[i].key;
	tmpid = Heaparray[i].run_id;   
	Heaparray[i].key = Heaparray[j].key;
	Heaparray[i].run_id = Heaparray[j].run_id;
	Heaparray[j].key = tmpkey;
	Heaparray[j].run_id = tmpid;
}

template<class REC, class KEY, class CMPR>
inline void merge_heap_dh_kop<REC,KEY,CMPR>::delete_min_and_insert
                                      (REC *nextelement_same_run)
{ 
  if (nextelement_same_run == NULL) {
    Heaparray[1].key = Heaparray[Heapsize].key;
    Heaparray[1].run_id = Heaparray[Heapsize].run_id;
    Heapsize--;
  } else { 
    UsrObject->copy(&Heaparray[1].key, *nextelement_same_run);
  }
  Heapify(1);
}

// Allocate space for the heap
template<class REC, class KEY, class CMPR>
inline void merge_heap_dh_kop<REC,KEY,CMPR>::allocate ( TPIE_OS_SIZE_T size ) {
    Heaparray = new heap_element<KEY> [size+1];
    Heapsize  = 0;
    maxHeapsize = size;
}

// Copy an (initial) element into the heap array
template<class REC, class KEY, class CMPR>
inline void merge_heap_dh_kop<REC,KEY,CMPR>::insert (REC *ptr,
                                                    TPIE_OS_SIZE_T run_id)
{
    UsrObject->copy(&Heaparray[Heapsize+1].key, *ptr);
    Heaparray[Heapsize+1].run_id = run_id;
    Heapsize++;
}

// Deallocate the space used by the heap
template<class REC, class KEY, class CMPR>
inline void merge_heap_dh_kop<REC,KEY,CMPR>::deallocate () {
  if (Heaparray){
    delete [] Heaparray; 
    Heaparray=NULL;
  }
  Heapsize    = 0;
  maxHeapsize = 0;
};

//Returns the index of the smallest element out of
//i, the left child of i, and the right child of i
template<class REC, class KEY, class CMPR>
inline TPIE_OS_SIZE_T merge_heap_dh_kop<REC,KEY,CMPR>::get_smallest(
                                                      TPIE_OS_SIZE_T i)
{
  TPIE_OS_SIZE_T l,r, smallest;
 
  l = Left(i);
  r = Right(i);
 
  smallest = ((l <= Heapsize) && 
      (Heaparray[l].key < Heaparray[i].key)) ? l : i;

  smallest = ((r <= Heapsize) && 
      (Heaparray[r].key < Heaparray[smallest].key))? r : smallest;

  return smallest;
}

// This is the primary function; note that we have unfolded the 
// recursion.
template<class REC, class KEY, class CMPR>
inline void merge_heap_dh_kop<REC,KEY,CMPR>::Heapify(TPIE_OS_SIZE_T i) {

  TPIE_OS_SIZE_T smallest=get_smallest(i);

  while (smallest != i) {
    this->Exchange(i,smallest);
    i = smallest;
    smallest = get_smallest(i);
  }
}

template<class REC, class KEY, class CMPR>
void merge_heap_dh_kop<REC,KEY,CMPR>::initialize ( ) {
    for ( TPIE_OS_SIZE_T i = Heapsize/2; i >= 1; i--) 
	this->Heapify(i);
}

// ********************************************************************
// * A key-merge heap that uses a comparison object                   *
// ********************************************************************

// The merge_heap_dh_kobj object maintains only the keys in its heap,
// and uses the member function "copy" of the user-provided class CMPR
// to copy these keys from each record. It uses the member function
// "compare" of the user-provided class CMPR to determine the relative
// order of two such keys in the sort order.

template<class REC, class KEY, class CMPR>
class merge_heap_dh_kobj: public merge_heap_dh_kop<REC,KEY,CMPR>{

protected: 
  
  using merge_heap_dh_kop<REC,KEY,CMPR>::Heapsize;
  using merge_heap_dh_kop<REC,KEY,CMPR>::Heaparray;
  using merge_heap_dh_kop<REC,KEY,CMPR>::maxHeapsize;
  using merge_heap_dh_kop<REC,KEY,CMPR>::UsrObject;
  
  inline TPIE_OS_SIZE_T get_smallest(TPIE_OS_SIZE_T i);

  inline void Heapify(TPIE_OS_SIZE_T);
public:
  using merge_heap_dh_kop<REC,KEY,CMPR>::sizeofheap;
  
  // Constructor initializes a pointer to the user's comparison object
  // The object may contain dynamic data although the 'compare' method is const
  // and therefore inline'able.
  merge_heap_dh_kobj ( CMPR *cmptr ) :
    merge_heap_dh_kop<REC, KEY, CMPR>(cmptr){};
  ~merge_heap_dh_kobj(){};
  
  // heapify's an initial array of elements
  void initialize (void);
  
  // Delete the current minimum and insert the new item from the same
  // source / run.
  inline void delete_min_and_insert(REC *nextelement_same_run);

};

//Returns the index of the smallest element out of
//i, the left child of i, and the right child of i
template<class REC, class KEY, class CMPR>
inline TPIE_OS_SIZE_T merge_heap_dh_kobj<REC,KEY,CMPR>::get_smallest(
                                                      TPIE_OS_SIZE_T i)
{
  TPIE_OS_SIZE_T l,r, smallest;
 
  l = Left(i);
  r = Right(i);
 
  smallest = ((l <= Heapsize) && 
      (UsrObject->compare(Heaparray[l].key,Heaparray[i].key)<0)) ? l : i;
  
  smallest = ((r <= Heapsize) && 
      (UsrObject->compare(Heaparray[r].key,Heaparray[smallest].key)<0)) ?
      r : smallest;

  return smallest;
}

template<class REC, class KEY, class CMPR>
inline void merge_heap_dh_kobj<REC, KEY, CMPR>::Heapify(TPIE_OS_SIZE_T i) {
    TPIE_OS_SIZE_T smallest = get_smallest(i);
    while (smallest != i) {
      this->Exchange(i,smallest);
      i = smallest;
      smallest = get_smallest(i);
    }
}

template<class REC, class KEY, class CMPR>
inline void merge_heap_dh_kobj<REC,KEY,CMPR>::delete_min_and_insert
                                      (REC *nextelement_same_run)
{ 
  if (nextelement_same_run == NULL) {
    Heaparray[1].key = Heaparray[Heapsize].key;
    Heaparray[1].run_id = Heaparray[Heapsize].run_id;
    Heapsize--;
  } else { 
    UsrObject->copy(&Heaparray[1].key, *nextelement_same_run);
  }
  Heapify(1);
}

template<class REC, class KEY, class CMPR>
void merge_heap_dh_kobj<REC, KEY, CMPR>::initialize () {
  for ( TPIE_OS_SIZE_T i = Heapsize/2; i >= 1; i--){ Heapify(i); }
}


#undef Left
#undef Right
#undef Parent

/*
   DEPRECATED: comparision function heaps
   Earlier TPIE versions allowed a heap that uses C-style
   comparison functions. However, comparison functions cannot be
   inlined, so each comparison requires one function call. Given that the
   comparison operator < and comparison object classes can be inlined and
   have better performance while providing the exact same functionality,
   comparison functions have been removed from TPIE. If you can provide us
   with a compelling argument on why they should be in here, we may consider
   adding them again, but you must demonstrate that comparision functions
   can outperform other methods in at least some cases or give an example
   were it is impossible to use a comparison operator or comparison object
*/

#endif // _MERGE_HEAP_DH_H
