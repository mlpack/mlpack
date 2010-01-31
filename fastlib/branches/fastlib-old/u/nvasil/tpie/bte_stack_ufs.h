//
// File:     bte_stack_ufs.h
// Author:   Octavian Procopiuc <tavi@cs.duke.edu>
// Created:  09/15/03
//
// A stack implemented using BTE_stream_ufs. It is used by
// BTE_collection_base to implement deletions.  
//
// $Id: bte_stack_ufs.h,v 1.2 2005/01/14 18:47:22 tavi Exp $
//

#ifndef _BTE_STACK_UFS_H
#define _BTE_STACK_UFS_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

#include "u/nvasil/tpie/bte_stream_ufs.h"

template<class T>
class BTE_stack_ufs : public BTE_stream_ufs<T> {
public:
  using BTE_stream_ufs<T>::stream_len;
  using BTE_stream_ufs<T>::seek;
  using BTE_stream_ufs<T>::truncate;
  
  // Construct a new stack with the given name and access type.
  BTE_stack_ufs(char *path, BTE_stream_type type = BTE_WRITE_STREAM); 
  // Destroy this object.
  ~BTE_stack_ufs(void);
  // Push an element on top of the stack.
  BTE_err push(const T &t);
  // Pop an element from the top of the stack.
  BTE_err pop(T **t);

};


template<class T>
BTE_stack_ufs<T>::BTE_stack_ufs(char *path, 
			    BTE_stream_type type) :
  BTE_stream_ufs<T>(path, type, 1)
{
}

template<class T>
BTE_stack_ufs<T>::~BTE_stack_ufs(void)
{
}

template<class T>
BTE_err BTE_stack_ufs<T>::push(const T &t)
{
  BTE_err ae;
  TPIE_OS_OFFSET slen;
    
  ae = truncate((slen = stream_len())+1);
  if (ae != BTE_ERROR_NO_ERROR) {
    return ae;
  }

  ae = seek(slen);
  if (ae != BTE_ERROR_NO_ERROR) {
    return ae;
  }

  return write_item(t);
}


template<class T>
BTE_err BTE_stack_ufs<T>::pop(T **t)
{
  BTE_err ae;
  TPIE_OS_OFFSET slen;
  
  slen = stream_len();
  ae = seek(slen-1);
  if (ae != BTE_ERROR_NO_ERROR) {
    return ae;
  }
  
  ae = read_item(t);
  if (ae != BTE_ERROR_NO_ERROR) {
    return ae;
  }

  return truncate(slen-1);
}

#endif // _BTE_STACK_UFS_H 
