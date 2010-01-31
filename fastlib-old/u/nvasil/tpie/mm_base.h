// Copyright (c) 1994 Darren Erik Vengroff
//
// File: mm_base.h
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/30/94
//
// $Id: mm_base.h,v 1.8 2005/04/22 13:52:50 jan Exp $
//
#ifndef _MM_BASE_H
#define _MM_BASE_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

#include <sys/types.h>

// dh. MM accounting modes
typedef enum {
  MM_IGNORE_MEMORY_EXCEEDED=0,
  MM_ABORT_ON_MEMORY_EXCEEDED,
  MM_WARN_ON_MEMORY_EXCEEDED
} MM_mode;

// MM Error codes
enum MM_err {
    MM_ERROR_NO_ERROR = 0,
    MM_ERROR_INSUFFICIENT_SPACE,
    MM_ERROR_UNDERFLOW,
    MM_ERROR_EXCESSIVE_ALLOCATION
};

// types of memory usage queries we can make on streams (either BTE or MM)
enum MM_stream_usage {
    // Overhead of the object without the buffer
    MM_STREAM_USAGE_OVERHEAD = 1,
    // Max amount ever used by a buffer
    MM_STREAM_USAGE_BUFFER,
    // Amount currently in use.
    MM_STREAM_USAGE_CURRENT,
    // Max amount that will ever be used.
    MM_STREAM_USAGE_MAXIMUM,
    // Maximum additional amount used by each substream created.
    MM_STREAM_USAGE_SUBSTREAM
};

// The base class for pointers into memory being managed by memory
// managers.  In a uniprocessor, these objects will simply contain
// pointers.  In multiprocessors, they will be more complicated
// descriptions of the layout of memory.

class MM_ptr_base
{
public:
    // This should return 1 to indicate a valid pointer and 0 to
    // indicate an invalid one.  It is useful for tests and
    // assertions.
    virtual operator int (void) = 0;
};

// The base class for all memory management objects.

class MM_manager_base
{
public:

    // How much is currently available.
    virtual MM_err available (TPIE_OS_SIZE_T *sz_a) = 0;

};

#endif // _MM_BASE_H 
