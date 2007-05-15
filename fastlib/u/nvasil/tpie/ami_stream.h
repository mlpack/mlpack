//
// File: ami_stream.h (formerly part of ami.h and ami_imps.h)
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
//
// $Id: ami_stream.h,v 1.6 2003/05/08 22:30:48 tavi Exp $
//
#ifndef _AMI_STREAM_H
#define _AMI_STREAM_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

#ifndef AMI_VIRTUAL_BASE
#  define AMI_VIRTUAL_BASE 0
#endif

// include definition of VERSION macro
#include "u/nvasil/tpie/versions.h"

// Include the configuration header.
#include "u/nvasil/tpie/config.h"

// Get the base class, enums, etc...
#include "u/nvasil/tpie/ami_err.h"
#include "u/nvasil/tpie/ami_stream_base.h"

// Get the device description class
#include "u/nvasil/tpie/ami_device.h"

// Get an implementation definition

#if defined(AMI_IMP_SINGLE)
	TPIE_OS_UNIX_ONLY_WARNING_AMI_IMP_SINGLE
#else
#  define AMI_STREAM_IMP_SINGLE
#endif

#if defined(AMI_IMP_USER_DEFINED)
#  warning The AMI_IMP_USER_DEFINED flag is obsolete. \
           Please use AMI_STREAM_IMP_USER_DEFINED.
#  warning Implicitly defining AMI_STREAM_IMP_USER_DEFINED.
#  define AMI_STREAM_IMP_USER_DEFINED
#endif

// The number of implementations to be defined.
#define _AMI_STREAM_IMP_COUNT (defined(AMI_STREAM_IMP_USER_DEFINED) +	\
		        defined(AMI_STREAM_IMP_SINGLE))

// Multiple implementations are allowed to coexist, with some
// restrictions.  Declarations of streams must use explicit subclasses
// of AMI_stream_base to specify what type of streams they are.

// If the including module did not explicitly ask for multiple
// implementations but requested more than one implementation, issue a
// warning.
#ifndef AMI_STREAM_IMP_MULTI_IMP
#  if (_AMI_STREAM_IMP_COUNT > 1)
#    warning Multiple AMI_STREAM_IMP_* defined, \
             but AMI_STREAM_IMP_MULTI_IMP undefined.
#    warning Implicitly defining AMI_STREAM_IMP_MULTI_IMP.
#    define AMI_STREAM_IMP_MULTI_IMP
#  endif // (_AMI_STREAM_IMP_COUNT > 1)
#endif // AMI_STREAM_IMP_MULTI_IMP

// If we have multiple implementations, set AMI_STREAM to be the base
// class.
#ifdef AMI_STREAM_IMP_MULTI_IMP
#  define AMI_STREAM AMI_stream_base
#  define AMI_stream AMI_stream_base
#endif

// Now include the definitions of each implementation that will be
// used.

// Make sure at least one implementation was chosen.  If none was,
// then choose one by default, but warn the user. [tavi] NO, don't
// bother. Since the IMP_SINGLE is the only existing implementation,
// just tacitly make it the default.
#if (_AMI_STREAM_IMP_COUNT < 1)
//#  warning No implementation defined. Using AMI_STREAM_IMP_SINGLE by default.
#  define AMI_STREAM_IMP_SINGLE
#endif // (_AMI_STREAM_IMP_COUNT < 1)

// User defined implementation.
#if defined(AMI_STREAM_IMP_USER_DEFINED)
   // Do nothing.  The user will provide a definition of AMI_STREAM.
#endif // defined(AMI_STREAM_IMP_USER_DEFINED)

// Single BTE stream implementation.
#if defined(AMI_STREAM_IMP_SINGLE)
#  include "u/nvasil/tpie/ami_stream_single.h"
   // If this is the only implementation, then make it easier to get to.
#  ifndef AMI_STREAM_IMP_MULTI_IMP
#    define AMI_STREAM AMI_stream_single
#    define AMI_stream AMI_stream_single
#  endif // AMI_STREAM_IMP_MULTI_IMP
#endif // defined(AMI_STREAM_IMP_SINGLE)

#endif // _AMI_STREAM_H
