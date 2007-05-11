// Copyright (c) 2005 Andrew Danner
//
// File: ami_queue.h
// Author: Andrew Danner <adanner@cs.duke.edu>
// Created: 2/22/05
//
// $Id: ami_queue.h,v 1.1 2005/04/25 19:08:06 adanner Exp $
//
#ifndef _AMI_QUEUE_H
#define _AMI_QUEUE_H

// Get definitions for working with Unix and Windows
#include <portability.h>
// Get the AMI_STREAM definition.
#include <ami_stream.h>
#include <ami_stack.h> 

// Basic Implementation of I/O Efficient FIFO queue. 
// Uses two stacks
template<class T>
class AMI_queue {

  public:
    bool empty();
    TPIE_OS_OFFSET size(){return Qsize;}
    AMI_queue(); 
    AMI_queue(const char* basename);
    ~AMI_queue(void);
    AMI_err enqueue(const T &t);
    AMI_err dequeue(T **t);
    void persist(persistence p);

  private:
    AMI_stack<T>* enQstack;
    AMI_stack<T>* deQstack;
    TPIE_OS_OFFSET Qsize;
};

//Constructor for Temporary Queue
template<class T>
AMI_queue<T>::AMI_queue() {
  enQstack = new AMI_stack<T>();
  deQstack = new AMI_stack<T>();
  enQstack->persist(PERSIST_DELETE);
  deQstack->persist(PERSIST_DELETE);
  Qsize=0;
}

//Constructor for Queue with filename
template<class T>
AMI_queue<T>::AMI_queue(const char* basename)
{
  char fname[BTE_STREAM_PATH_NAME_LEN];
  strncpy(fname, basename, BTE_STREAM_PATH_NAME_LEN-4);
  strcat(fname,".nq"); 
  enQstack = new AMI_stack<T>(fname);
  strncpy(fname, basename, BTE_STREAM_PATH_NAME_LEN-4);
  strcat(fname,".dq"); 
  deQstack = new AMI_stack<T>(fname);
  enQstack->persist(PERSIST_PERSISTENT);
  deQstack->persist(PERSIST_PERSISTENT);
  Qsize=enQstack->stream_len()+deQstack->stream_len();
}

template<class T>
AMI_queue<T>::~AMI_queue(void)
{
  delete enQstack;
  delete deQstack;
}

template<class T>
void AMI_queue<T>::persist(persistence p) {
  enQstack->persist(p);
  deQstack->persist(p);
}

template<class T>
AMI_err AMI_queue<T>::enqueue(const T &t)
{
  //Elements are pushed onto an Enqueue stack
  AMI_err ae=enQstack->push(t);
  if(ae == AMI_ERROR_NO_ERROR){
    Qsize++;
  }
  return ae;
}

template<class T>
AMI_err AMI_queue<T>::dequeue(T **t)
{
    AMI_err ae;
    T* tmp;
    //Elements popped from Dequeue stack
    if(deQstack->stream_len()>0){
      ae=deQstack->pop(t);
      if(ae == AMI_ERROR_NO_ERROR){
        Qsize--;
      }
      return ae;
    }
    else if(Qsize == 0){
      return AMI_ERROR_END_OF_STREAM;
    }
    else{
      //move elements from Enqueue stack to Dequeue stack
      while((ae=enQstack->pop(&tmp)) == AMI_ERROR_NO_ERROR){
        ae=deQstack->push(*tmp);
        if(ae != AMI_ERROR_NO_ERROR){ return ae; }
      }
      if(ae != AMI_ERROR_BTE_ERROR){
        return ae;
      }
      ae=deQstack->pop(t);
      if(ae == AMI_ERROR_NO_ERROR){
        Qsize--;
      }
      return ae;
    }
}

#endif // _AMI_QUEUE_H 
