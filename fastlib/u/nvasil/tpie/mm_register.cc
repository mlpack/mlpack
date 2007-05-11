// Copyright (c) 1994 Darren Erik Vengroff
//
// File: mm_register.cpp
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/31/94
//

// A simple registration based memory manager.

#include <versions.h>
VERSION(mm_register_cpp,"$Id: mm_register.cpp,v 1.23 2005/07/07 20:37:39 adanner Exp $");

//#include <assert.h>
#include "lib_config.h"

#define MM_IMP_REGISTER
#include <mm.h>
#include <mm_register.h>

#ifdef REPORT_LARGE_MEMOPS
#include <iostream>
#endif

#ifdef MM_BACKWARD_COMPATIBLE
extern int register_new;
#endif

#include <stdlib.h>

MM_register::MM_register()
{
    instances++;

    tp_assert(instances == 1,
              "Only 1 instance of MM_register_base should exist.");
}
 

MM_register::~MM_register(void)
{
    tp_assert(instances == 1,
              "Only 1 instance of MM_register_base should exist.");

    instances--;
}

// check that new allocation request is below user-defined limit.
// This should be a private method, only called by operator new.

MM_err MM_register::register_allocation(TPIE_OS_SIZE_T request)
{
  // quick hack to allow operation before limit is set
  // XXX 
  if(!user_limit) {
	return MM_ERROR_NO_ERROR;
  }

    used      += request;     

    if (request > remaining) {
       TP_LOG_WARNING("Memory allocation request: ");
       TP_LOG_WARNING(static_cast<TPIE_OS_OUTPUT_SIZE_T>(request));
       TP_LOG_WARNING(": User-specified memory limit exceeded.");
       TP_LOG_FLUSH_LOG;
       remaining = 0;
       return MM_ERROR_INSUFFICIENT_SPACE;
    }

    remaining -= request; 

    TP_LOG_MEM_DEBUG("mm_register Allocated ");
    TP_LOG_MEM_DEBUG(static_cast<TPIE_OS_OUTPUT_SIZE_T>(request));
    TP_LOG_MEM_DEBUG("; ");
    TP_LOG_MEM_DEBUG(static_cast<TPIE_OS_OUTPUT_SIZE_T>(remaining));
    TP_LOG_MEM_DEBUG(" remaining.\n");
    TP_LOG_FLUSH_LOG;

#ifdef REPORT_LARGE_MEMOPS
	if(request > user_limit/10) {
	  cerr << "MEM alloc " << request
		   << " (" << remaining << " remaining)" << endl;
	}
#endif
    
    return MM_ERROR_NO_ERROR;
}

// do the accounting for a memory deallocation request.
// This should be a private method, only called by operators 
// delete and delete [].

MM_err MM_register::register_deallocation(TPIE_OS_SIZE_T sz)
{
    remaining += sz;

    if (sz > used) {
       TP_LOG_WARNING("Error in deallocation sz=");
       TP_LOG_WARNING((TPIE_OS_LONG)sz);
       TP_LOG_WARNING(", remaining=");
       TP_LOG_WARNING((TPIE_OS_LONG)remaining);
       TP_LOG_WARNING(", user_limit=");
       TP_LOG_WARNING((TPIE_OS_LONG)user_limit);
       TP_LOG_WARNING("\n");
       TP_LOG_FLUSH_LOG;
       used = 0;
       return MM_ERROR_UNDERFLOW;
    }

    used      -= sz;    

    TP_LOG_MEM_DEBUG("mm_register De-allocated ");
    TP_LOG_MEM_DEBUG((unsigned int)sz);
    TP_LOG_MEM_DEBUG("; ");
    TP_LOG_MEM_DEBUG((unsigned int)remaining);
    TP_LOG_MEM_DEBUG(" now available.\n");
    TP_LOG_FLUSH_LOG;
    
#ifdef REPORT_LARGE_MEMOPS
	if(sz > user_limit/10) {
	  cerr << "MEM free " << sz 
		   << " (" << remaining << " remaining)" << endl;
	}
#endif

    return MM_ERROR_NO_ERROR;
}

#ifdef MM_BACKWARD_COMPATIBLE
// (Old) way to query how much memory is available

MM_err MM_register::available (TPIE_OS_SIZE_T *sz)
{
    *sz = remaining;
    return MM_ERROR_NO_ERROR;    
}

// resize_heap has the same purpose as set_memory_limit.
// It is retained for backward compatibility. 
// dh. 1999 09 29

MM_err MM_register::resize_heap(TPIE_OS_SIZE_T sz)
{
   return set_memory_limit(sz);
}
#endif


// User-callable method to set allowable memory size

MM_err MM_register::set_memory_limit (TPIE_OS_SIZE_T new_limit)
{
    // by default, we keep track and abort if memory limit exceeded
    if (register_new == MM_IGNORE_MEMORY_EXCEEDED){
       register_new = MM_ABORT_ON_MEMORY_EXCEEDED;
    }
    // dh. unless the user indicates otherwise
    if (new_limit == 0){
       register_new = MM_IGNORE_MEMORY_EXCEEDED;
       remaining = used = user_limit = 0;
       return MM_ERROR_NO_ERROR;
    } 

    if (used > new_limit) {
        return MM_ERROR_EXCESSIVE_ALLOCATION;
    } else {
        // These are unsigned, so be careful.
        if (new_limit < user_limit) {
            remaining -= user_limit - new_limit;
        } else {
            remaining += new_limit - user_limit;
        }
        user_limit = new_limit;
        return MM_ERROR_NO_ERROR;
    }
}

// dh. only warn if memory limit exceeded
void MM_register::warn_memory_limit()
{
    register_new = MM_WARN_ON_MEMORY_EXCEEDED;
}

// dh. abort if memory limit exceeded
void MM_register::enforce_memory_limit()
{
    register_new = MM_ABORT_ON_MEMORY_EXCEEDED;
}

// dh. ignore memory limit accounting
void MM_register::ignore_memory_limit()
{
    register_new = MM_IGNORE_MEMORY_EXCEEDED;
}

// rw. provide accounting state
MM_mode MM_register::get_limit_mode() {
  return register_new;
}


// dh. return the amount of memory available before user-specified 
// memory limit exceeded 
TPIE_OS_SIZE_T MM_register::memory_available()
{
    return remaining;    
}

size_t MM_register::memory_used()
{
    return used;    
}

size_t MM_register::memory_limit()
{
    return user_limit;    
}

// Instantiate the actual memory manager, and allocate the 
// its static data members
MM_register MM_manager;
int MM_register::instances = 0; // Number of instances. (init)
// TPIE's "register memory requests" flag
MM_mode MM_register::register_new = MM_ABORT_ON_MEMORY_EXCEEDED; 

// The counter of mm_register_init instances.  It is implicity set to 0.
unsigned int mm_register_init::count;

// The constructor and destructor that ensure that the memory manager is
// created exactly once, and destroyed when appropriate.
mm_register_init::mm_register_init(void)
{
    if (count++ == 0) {
	  MM_manager.set_memory_limit(MM_DEFAULT_MM_SIZE);
    }
}

mm_register_init::~mm_register_init(void)
{
    --count;
}
