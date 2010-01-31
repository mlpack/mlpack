//
// File:   ami_coll.h
// Author: Octavian Procopiuc <tavi@cs.duke.edu>
//
// $Id: ami_coll.h,v 1.8 2003/05/08 22:12:21 tavi Exp $
//
// Front end for the AMI_COLLECTION implementations.
//
#ifndef _AMI_COLL_H
#define _AMI_COLL_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

#include "u/nvasil/tpie/ami_coll_base.h"
#include "u/nvasil/tpie/ami_coll_single.h"

// AMI_collection_single is the only implementation, so make it easy
// to get to.

#define AMI_collection AMI_collection_single

#ifdef BTE_COLLECTION
#  define AMI_COLLECTION AMI_collection_single< BTE_COLLECTION >
#endif

#endif // _AMI_COLL_H
