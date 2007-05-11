//
// File:	stdio_stack.h
// Authors:	Rakesh Barve
//
// Defining a stack based on bte_stdio separately specifically
// for use in block collection class and related apps.
// The reason we don't want to use ami_stack is because 
// then the stack would be implemented as a BTE_STREAM, which
// may have large block size etc. which is undesirable for 
// stacks related to block collections since such a stack is only
// a meta data structure accessed no more than once every block
// is created or destroyed. 
//
// $Id: stdio_stack.h,v 1.7 2003/04/17 19:57:25 jan Exp $
//

#ifndef _STDIO_STACK_H
#define _STDIO_STACK_H

// Get definitions for working with Unix and Windows
#include <portability.h>

#include <bte_stream_stdio.h>

template<class T>
class stdio_stack : public BTE_stream_stdio<T> {
public:

  stdio_stack(char *path, BTE_stream_type type = BTE_WRITE_STREAM); 

  ~stdio_stack(void);

  BTE_err push(const T &t);

  BTE_err pop(T **t);

};


template<class T>
stdio_stack<T>::stdio_stack(char *path, 
			    BTE_stream_type type) :
  BTE_stream_stdio<T>(path, type)
{
}

template<class T>
stdio_stack<T>::~stdio_stack(void)
{
}

template<class T>
BTE_err stdio_stack<T>::push(const T &t)
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
BTE_err stdio_stack<T>::pop(T **t)
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

#endif // _stdio_stack_H 
