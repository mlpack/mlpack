// Copyright (c) 1994 Darren Erik Vengroff
//
// File: mm_base.cpp
// Author: Darren Erik Vengroff <darrenv@eecs.umich.edu>
// Created: 9/2/94
//

#include <versions.h>
VERSION(mm_base_cpp,"$Id: mm_base.cpp,v 1.29 2004/10/27 19:13:54 adanner Exp $");

#include "lib_config.h"
#include <mm_base.h>
#include <tpie_log.h>
#include <mm_register.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// support for dmalloc (for tracking memory leaks)
#ifdef USE_DMALLOC
#define DMALLOC_DISABLE
#include <dmalloc.h>
#include <return.h>
#endif

#ifdef MM_BACKWARD_COMPATIBLE
int   register_new = MM_IGNORE_MEMORY_EXCEEDED;
#endif

// SIZE_SPACE is to ensure alignment on quad word boundaries.  It may be
// possible to check whether a machine needs this at configuration
// time or if dword alignment is ok.  On the HP 9000, bus errors occur
// when loading doubles that are not qword aligned.
static const TPIE_OS_SIZE_T SIZE_SPACE=(sizeof(TPIE_OS_SIZE_T) > 8 ? sizeof(TPIE_OS_SIZE_T) : 8);

void *operator new (TPIE_OS_SIZE_T sz)
{
   void *p;
#ifdef USE_DMALLOC
	char	*file;
	GET_RET_ADDR(file);
#endif

   if ((MM_manager.register_new != MM_IGNORE_MEMORY_EXCEEDED)
       && (MM_manager.register_allocation (sz + SIZE_SPACE) !=
	   MM_ERROR_NO_ERROR)) {
      switch(MM_manager.register_new) {
	  case MM_ABORT_ON_MEMORY_EXCEEDED:
		TP_LOG_FATAL_ID ("In operator new() - allocation request \"");
		TP_LOG_FATAL ((TPIE_OS_LONG)(sz + SIZE_SPACE));
		TP_LOG_FATAL ("\" plus previous allocation \"");
		TP_LOG_FATAL ((TPIE_OS_LONG)(MM_manager.memory_used () - (sz + SIZE_SPACE)));
		TP_LOG_FATAL ("\" exceeds user-defined limit \"");
		TP_LOG_FATAL ((TPIE_OS_LONG)(MM_manager.memory_limit ()));
		TP_LOG_FATAL ("\" \n");
		TP_LOG_FLUSH_LOG;
		cerr << "memory manager: memory allocation limit " << 
                         (TPIE_OS_LONG)MM_manager.memory_limit () << " exceeded "
			 << "while allocating " << (TPIE_OS_LONG)sz << " bytes" << "\n";
#ifdef USE_DMALLOC
		dmalloc_shutdown();
#endif
		assert (0);		// core dump if debugging
		exit (1);
		break;
	  case MM_WARN_ON_MEMORY_EXCEEDED:
		TP_LOG_WARNING_ID ("In operator new() - allocation request \"");
		TP_LOG_WARNING ((TPIE_OS_LONG)(sz + SIZE_SPACE));
		TP_LOG_WARNING ("\" plus previous allocation \"");
		TP_LOG_WARNING ((TPIE_OS_LONG)(MM_manager.memory_used () - (sz + SIZE_SPACE)));
		TP_LOG_WARNING ("\" exceeds user-defined limit \"");
		TP_LOG_WARNING ((TPIE_OS_LONG)(MM_manager.memory_limit ()));
		TP_LOG_WARNING ("\" \n");
		TP_LOG_FLUSH_LOG;
		cerr << "memory manager: memory allocation limit " << 
                         (TPIE_OS_LONG)MM_manager.memory_limit () << " exceeded "
			 << "while allocating " << (TPIE_OS_LONG)sz << " bytes" << "\n";
		break;
	  case MM_IGNORE_MEMORY_EXCEEDED:
		break;
      }
   }

#ifdef USE_DMALLOC
   p = _malloc_leap(file, 0, sz + SIZE_SPACE);
#else
   p = malloc(sz + SIZE_SPACE);
#endif
   if (!p) {
      TP_LOG_FATAL_ID ("Out of memory. Cannot continue.");
      TP_LOG_FLUSH_LOG;
	  cerr << "out of memory while allocating " << (TPIE_OS_LONG)sz << " bytes" << "\n";
      perror ("mm_base::new malloc");
	  assert(0);
      exit (1);
   }
   *((size_t *) p) = sz;
   return ((char *) p) + SIZE_SPACE;
}

void operator delete (void *ptr)
{
   if (!ptr) {
      TP_LOG_WARNING_ID ("operator delete was given a NULL pointer");
      return;
   }

   if (MM_manager.register_new != MM_IGNORE_MEMORY_EXCEEDED) {
      if (MM_manager.register_deallocation (
	    (TPIE_OS_SIZE_T)*((size_t *) (((char *) ptr) - SIZE_SPACE)) + SIZE_SPACE) 
         != MM_ERROR_NO_ERROR) {
	 TP_LOG_WARNING_ID("In operator delete - MM_manager.register_deallocation failed");
      }
   }
   void *p = ((char *)ptr) - SIZE_SPACE;

#ifdef USE_DMALLOC
	char	*file;
	GET_RET_ADDR(file);
	_free_leap(file, 0, p);    
#else
    free(p);
#endif
}

void operator delete[] (void *ptr) {
   if (!ptr) {
      TP_LOG_WARNING_ID ("operator delete [] was given a NULL pointer");
      return;
   }

   if (MM_manager.register_new != MM_IGNORE_MEMORY_EXCEEDED) {
      if (MM_manager.register_deallocation (
	    (TPIE_OS_SIZE_T)*((size_t *) (((char *) ptr) - SIZE_SPACE)) + SIZE_SPACE)
         != MM_ERROR_NO_ERROR) {
	 TP_LOG_WARNING_ID("In operator delete [] - MM_manager.register_deallocation failed");
      }
   }
   void *p = ((char *)ptr) - SIZE_SPACE;
#ifdef USE_DMALLOC
	char	*file;
	GET_RET_ADDR(file);
	_free_leap(file, 0, p);    
#else
    free(p);
#endif
}

// return the overhead on each memory allocation request 
int   MM_register::space_overhead ()
{
   return SIZE_SPACE;
}

#ifndef NDEBUG
TPIE_OS_SPACE_OVERHEAD_BODY
#endif

