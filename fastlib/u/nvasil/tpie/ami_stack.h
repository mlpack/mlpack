// Copyright (c) 1994 Darren Vengroff
//
// File: ami_stack.h
// Author: Darren Vengroff <darrenv@eecs.umich.edu>
// Created: 12/15/94
//
// $Id: ami_stack.h,v 1.9 2005/01/21 16:52:39 tavi Exp $
//
#ifndef _AMI_STACK_H
#define _AMI_STACK_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"
// Get the AMI_STREAM definition.
#include "u/nvasil/tpie/ami_stream.h"

template<class T>
class AMI_stack : public AMI_STREAM<T> {
  public:
    using AMI_STREAM<T>::seek;
    using AMI_STREAM<T>::truncate;
    using AMI_STREAM<T>::stream_len;

    AMI_stack(); 
    AMI_stack(const char* path, 
        AMI_stream_type type = AMI_READ_WRITE_STREAM);
    ~AMI_stack(void);
    AMI_err push(const T &t);
    AMI_err pop(T **t);
};


template<class T>
AMI_stack<T>::AMI_stack() :
        AMI_STREAM<T>()
{
}

template<class T>
AMI_stack<T>::AMI_stack(const char* path, AMI_stream_type type):
        AMI_STREAM<T>(path, type)
{
}

template<class T>
AMI_stack<T>::~AMI_stack(void)
{
}

template<class T>
AMI_err AMI_stack<T>::push(const T &t)
{
    AMI_err ae;
    TPIE_OS_OFFSET slen;
    
    ae = truncate((slen = stream_len())+1);
    if (ae != AMI_ERROR_NO_ERROR) {
        return ae;
    }

    ae = seek(slen);
    if (ae != AMI_ERROR_NO_ERROR) {
        return ae;
    }

    return write_item(t);
}


template<class T>
AMI_err AMI_stack<T>::pop(T **t)
{
    AMI_err ae;
    TPIE_OS_OFFSET slen;

    slen = stream_len();

    ae = seek(slen-1);
    if (ae != AMI_ERROR_NO_ERROR) {
        return ae;
    }

    ae = read_item(t);
    if (ae != AMI_ERROR_NO_ERROR) {
        return ae;
    }

    return truncate(slen-1);
}

#endif // _AMI_STACK_H 
