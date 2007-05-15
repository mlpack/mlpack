// Copyright (c) 2005 Andrew Danner
//
// File: internal_sorter.h
// Author: Andrew Danner <adanner@cs.duke.edu>
// Created: 28 Jun 2005
//
// Internal sorter class that can be used within AMI_sort() on small
// streams/substreams
//
// $Id: internal_sort.h,v 1.1 2005/08/24 19:32:38 adanner Exp $
//
#ifndef _INTERNAL_SORT_H
#define _INTERNAL_SORT_H
 
// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"
#include "u/nvasil/tpie/quicksort.h"

// Use our quicksort, or the sort from STL
#ifdef TPIE_USE_STL_SORT
// portability.h includes <algorithm> for us in the case of STL sort
#include <comparator.h> //to convert TPIE comparisons to STL
#endif

// The base class. This class does not have a sort() function, so it
// cannot be used directly
template<class T>
class Internal_Sorter_Base{
  protected:
    T* ItemArray;        //Array that holds items to be sorted
    TPIE_OS_OFFSET len;  //length of ItemArray

  public:
    //Constructor
    Internal_Sorter_Base(void): len(0) { ItemArray=NULL; }
    ~Internal_Sorter_Base(void); //Destructor
    
    //Allocate array that can hold nItems
    void allocate(TPIE_OS_OFFSET nItems);
    
    void deallocate(void); //Clean up internal array

    // Maximum number of Items that can be sorted using memSize bytes
    TPIE_OS_OFFSET MaxItemCount(TPIE_OS_OFFSET memSize);

    // Memory usage per sort item
    TPIE_OS_SIZE_T space_per_item();
    
    // Fixed memory usage overhead per class instantiation
    TPIE_OS_SIZE_T space_overhead();
};

template<class T>
Internal_Sorter_Base<T>::~Internal_Sorter_Base(void){
    //In case someone forgot to call deallocate()
    if(ItemArray){
      delete [] ItemArray;
      ItemArray=NULL;
    }
}

template<class T>
inline void Internal_Sorter_Base<T>::allocate(TPIE_OS_OFFSET nitems){
  len=nitems;
  ItemArray = new T[len];
}

template<class T>
inline void Internal_Sorter_Base<T>::deallocate(void){
  if(ItemArray){
    delete [] ItemArray;
    ItemArray=NULL;
    len=0;
  }
}

template<class T>
inline TPIE_OS_OFFSET Internal_Sorter_Base<T>::MaxItemCount(
                                              TPIE_OS_OFFSET memSize)
{
  //Space available for items
  TPIE_OS_OFFSET memAvail=memSize-space_overhead();
  
  if(memAvail < space_per_item() ){ return -1; }
  else{ return memAvail/space_per_item(); }
}


template<class T>
inline TPIE_OS_SIZE_T Internal_Sorter_Base<T>::space_overhead(void)
{ 
  // Space usage independent of space_per_item
  // accounts MM_manager space overhead on "new" call
  return MM_manager.space_overhead();
}

template<class T>
inline TPIE_OS_SIZE_T Internal_Sorter_Base<T>::space_per_item(void)
{
  return sizeof(T);
}

// *********************************************************************
// *                                                                   *
// * Operator based Internal Sorter.                                   *
// *                                                                   *
// *********************************************************************

template<class T>
class Internal_Sorter_Op: public Internal_Sorter_Base<T>{
  protected:
    using Internal_Sorter_Base<T>::len;
    using Internal_Sorter_Base<T>::ItemArray;

  public:
    //Constructor/Destructor
    Internal_Sorter_Op(){};
    ~Internal_Sorter_Op(){};
    
    using Internal_Sorter_Base<T>::space_overhead;

    //Sort nItems from input stream and write to output stream
    AMI_err sort(AMI_STREAM<T>* InStr, AMI_STREAM<T>* OutStr,
        TPIE_OS_OFFSET nItems);
};

// Read nItems sequentially from InStr, starting at the current file
// position. Write the sorted output to OutStr, starting from the current
// file position.
template<class T>
AMI_err Internal_Sorter_Op<T>::sort(AMI_STREAM<T>* InStr,
                AMI_STREAM<T>* OutStr, TPIE_OS_OFFSET nItems){

  AMI_err ae;
  T    *next_item;
  TPIE_OS_OFFSET i = 0;

  TP_LOG_DEBUG_ID("Sorting internal run of " << nItems 
                  << " items using \"<\" operator.");
  tp_assert ( nItems <= len, "nItems more than interal buffer size.");

  // Read a memory load out of the input stream one item at a time,
  for (i = 0; i < nItems; i++) {
    if ((ae=InStr->read_item (&next_item)) != AMI_ERROR_NO_ERROR) {
      TP_LOG_FATAL_ID ("Internal sort: AMI read error " << ae);
      return ae;
    }
    ItemArray[i] = *next_item;
  }

  //Sort the array.
#ifdef TPIE_USE_STL_SORT
  TP_LOG_DEBUG_ID("calling STL sort for " << nItems << " items");
  std::sort(ItemArray, ItemArray+nItems);
#else 
  TP_LOG_DEBUG_ID("calling quick_sort_op for " << nItems << " items");
  quick_sort_op<T> (ItemArray, nItems);
#endif

  if(InStr==OutStr){ //Do the right thing if we are doing 2x sort
    //Internal sort objects should probably be re-written so that
    //the interface is cleaner and they don't have to worry about I/O
    InStr->truncate(0); //delete original items
    InStr->seek(0); //rewind
  }
  //Write sorted array to OutStr
  for (i = 0; i < nItems; i++) {
    if ((ae = OutStr->write_item(ItemArray[i])) != AMI_ERROR_NO_ERROR) {
      TP_LOG_FATAL_ID ("Internal Sorter: AMI write error" << ae );
      return ae;
    }
  }
  TP_LOG_DEBUG_ID("returning from Internal_Sorter_Op");
  return AMI_ERROR_NO_ERROR;
}

// *********************************************************************
// *                                                                   *
// * Comparison object based Internal Sorter.                          *
// *                                                                   *
// *********************************************************************

template<class T, class CMPR>
class Internal_Sorter_Obj: public Internal_Sorter_Base<T>{
  protected:
    using Internal_Sorter_Base<T>::ItemArray;
    using Internal_Sorter_Base<T>::len;
    CMPR *cmp_o; //Comparison object used for sorting
    
  public:
    //Constructor/Destructor
    Internal_Sorter_Obj(CMPR* cmp){cmp_o=cmp;};
    ~Internal_Sorter_Obj(){};
    
    using Internal_Sorter_Base<T>::space_overhead;

    //Sort nItems from input stream and write to output stream
    AMI_err sort(AMI_STREAM<T>* InStr, AMI_STREAM<T>* OutStr,
        TPIE_OS_OFFSET nItems);
};

// Read nItems sequentially from InStr, starting at the current file
// position. Write the sorted output to OutStr, starting from the current
// file position.
template<class T, class CMPR>
AMI_err Internal_Sorter_Obj<T, CMPR>::sort(AMI_STREAM<T>* InStr,
                AMI_STREAM<T>* OutStr, TPIE_OS_OFFSET nItems){

  AMI_err ae;
  T    *next_item;
  TPIE_OS_OFFSET i = 0;

  TP_LOG_DEBUG_ID("Sorting internal run of " << nItems 
                  << " items using TPIE comparison object.");
  tp_assert ( nItems <= len, "nItems more than interal buffer size.");

  // Read a memory load out of the input stream one item at a time,
  for (i = 0; i < nItems; i++) {
    if ((ae=InStr->read_item (&next_item)) != AMI_ERROR_NO_ERROR) {
      TP_LOG_FATAL_ID ("Internal sort: AMI read error " << ae);
      return ae;
    }
    ItemArray[i] = *next_item;
  }

  //Sort the array.
#ifdef TPIE_USE_STL_SORT
  TP_LOG_DEBUG_ID("calling STL sort for " << nItems << " items");
  TP_LOG_DEBUG_ID("converting TPIE comparison object to STL");
  std::sort(ItemArray, ItemArray+nItems, TPIE2STL_cmp<T,CMPR>(cmp_o));
#else 
  TP_LOG_DEBUG_ID("calling quick_sort_obj for " << nItems << " items");
  quick_sort_obj<T> (ItemArray, nItems, cmp_o);
#endif

  if(InStr==OutStr){ //Do the right thing if we are doing 2x sort
    //Internal sort objects should probably be re-written so that
    //the interface is cleaner and they don't have to worry about I/O
    InStr->truncate(0); //delete original items
    InStr->seek(0); //rewind
  }
  //Write sorted array to OutStr
  for (i = 0; i < nItems; i++) {
    if ((ae = OutStr->write_item(ItemArray[i])) != AMI_ERROR_NO_ERROR) {
      TP_LOG_FATAL_ID ("Internal Sorter: AMI write error" << ae );
      return ae;
    }
  }
  TP_LOG_DEBUG_ID("returning from Internal_Sorter_Op");
  return AMI_ERROR_NO_ERROR;
}

// *********************************************************************
// *                                                                   *
// * Key + Object based Internal Sorter                                *
// *                                                                   *
// *********************************************************************

template<class T, class KEY, class CMPR>
class Internal_Sorter_KObj{
  protected:
    T* ItemArray;                    //Array that holds original items
    qsort_item<KEY>* sortItemArray;  //Holds keys to be sorted
    CMPR *UsrObject;                 //Copy,compare keys
    TPIE_OS_OFFSET len;              //length of ItemArray

  public:
    //Constructor:
    Internal_Sorter_KObj(CMPR* cmp): len(0) {
      ItemArray=NULL;
      sortItemArray=NULL;
      UsrObject=cmp;
    }
    ~Internal_Sorter_KObj(void); //Destructor
    
    //Allocate array that can hold nItems
    void allocate(TPIE_OS_OFFSET nItems);
    
    //Sort nItems from input stream and write to output stream
    AMI_err sort(AMI_STREAM<T>* InStr, AMI_STREAM<T>* OutStr,
        TPIE_OS_OFFSET nItems);
    
    void deallocate(void); //Clean up internal array

    // Maximum number of Items that can be sorted using memSize bytes
    TPIE_OS_OFFSET MaxItemCount(TPIE_OS_OFFSET memSize);

    // Memory usage per sort item
    TPIE_OS_SIZE_T space_per_item();
    
    // Fixed memory usage overhead per class instantiation
    TPIE_OS_SIZE_T space_overhead();
};

template<class T, class KEY, class CMPR>
Internal_Sorter_KObj<T, KEY, CMPR>::~Internal_Sorter_KObj(void){
    //In case someone forgot to call deallocate()
    if(ItemArray){
      delete [] ItemArray;
      ItemArray=NULL;
    }
    if(sortItemArray){
      delete [] sortItemArray;
      sortItemArray=NULL;
    }
}

template<class T, class KEY, class CMPR>
inline void Internal_Sorter_KObj<T, KEY, CMPR>::allocate(TPIE_OS_OFFSET nitems){
  len=nitems;
  ItemArray = new T[len];
  sortItemArray = new qsort_item<KEY>[len];
}

// A helper class to quick sort qsort_item<KEY> types
// given a comparison object for comparing keys
template<class KEY, class KCMP>
class QsortKeyCmp{
  private:
     KCMP *isLess; //Class with function compare that compares 2 keys

  public:
     QsortKeyCmp(KCMP* kcmp) {isLess=kcmp; }
     inline int compare(const qsort_item<KEY>& left,
                        const qsort_item<KEY>& right){
       return isLess->compare(left.keyval, right.keyval);
     }
};

template<class T, class KEY, class CMPR>
inline AMI_err Internal_Sorter_KObj<T, KEY, CMPR>::sort(AMI_STREAM<T>* InStr,
                AMI_STREAM<T>* OutStr, TPIE_OS_OFFSET nItems){

  AMI_err ae;
  T    *next_item;
  TPIE_OS_OFFSET i = 0;

  TP_LOG_DEBUG_ID("Sorting internal run of " << nItems 
                  << " items using \"<\" operator.");
  tp_assert ( nItems <= len, "nItems more than interal buffer size.");

  // Read a memory load out of the input stream one item at a time,
  for (i = 0; i < nItems; i++) {
    if ((ae=InStr->read_item (&next_item)) != AMI_ERROR_NO_ERROR) {
      TP_LOG_FATAL_ID ("Internal sort: AMI read error " << ae);
      return ae;
    }
    ItemArray[i] = *next_item;
    UsrObject->copy(&sortItemArray[i].keyval, *next_item);
    sortItemArray[i].source=i;
  }

  //Sort the array.
#ifdef TPIE_USE_STL_SORT
  TP_LOG_DEBUG_ID("calling STL sort for " << nItems << " items");
  TP_LOG_DEBUG_ID("converting TPIE Key comparison object to STL");
  std::sort(sortItemArray, ItemArray+nItems, 
      TPIE2STL_cmp<qsort_item<KEY>,QsortKeyCmp<KEY,CMPR> >
      (QsortKeyCmp<KEY, CMPR>(UsrObject)));
#else
  QsortKeyCmp<KEY, CMPR> qcmp(UsrObject);
 TP_LOG_DEBUG_ID("calling quick_sort_obj for " << nItems << " items");
  quick_sort_obj< qsort_item<KEY> > (sortItemArray, nItems, &qcmp);
#endif
  
  if(InStr==OutStr){ //Do the right thing if we are doing 2x sort
    //Internal sort objects should probably be re-written so that
    //the interface is cleaner and they don't have to worry about I/O
    InStr->truncate(0); //delete original items
    InStr->seek(0); //rewind
  }
  //Write sorted array to OutStr
  for (i = 0; i < nItems; i++) {
    if ((ae = OutStr->write_item(ItemArray[sortItemArray[i].source]))
        != AMI_ERROR_NO_ERROR) {
      TP_LOG_FATAL_ID ("Internal Sorter: AMI write error" << ae );
      return ae;
    }
  }
  TP_LOG_DEBUG_ID("returning from Internal_Sorter_Op");
  return AMI_ERROR_NO_ERROR;
}

template<class T, class KEY, class CMPR>
inline void Internal_Sorter_KObj<T, KEY, CMPR>::deallocate(void){
  len=0;
  if(ItemArray){
    delete [] ItemArray;
    ItemArray=NULL;
  }
  if(sortItemArray){
    delete [] sortItemArray;
    sortItemArray=NULL;
  }
}

template<class T, class KEY, class CMPR>
inline TPIE_OS_OFFSET Internal_Sorter_KObj<T, KEY, CMPR>::MaxItemCount(
                                              TPIE_OS_OFFSET memSize)
{
  //Space available for items
  TPIE_OS_OFFSET memAvail=memSize-space_overhead();
  
  if(memAvail < space_per_item() ){ return -1; }
  else{ return memAvail/space_per_item(); }
}


template<class T, class KEY, class CMPR>
inline TPIE_OS_SIZE_T Internal_Sorter_KObj<T, KEY, CMPR>::space_overhead(void)
{ 
  // Space usage independent of space_per_item
  // accounts MM_manager space overhead on "new" call
  return 2*MM_manager.space_overhead();
}

template<class T, class KEY, class CMPR>
inline TPIE_OS_SIZE_T Internal_Sorter_KObj<T, KEY, CMPR>::space_per_item(void)
{
  return sizeof(T) + sizeof(qsort_item<KEY>);
}

#endif // _INTERNAL_SORT_H 














































