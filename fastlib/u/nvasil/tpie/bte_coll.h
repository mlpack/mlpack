//
// File:    bte_coll.h
// Authors: Octavian Procopiuc <tavi@cs.duke.edu>
//
// $Id: bte_coll.h,v 1.4 2003/04/29 05:29:42 tavi Exp $
//
// Front end for the BTE collection classes.
//

#ifndef _BTE_COLL_H
#define _BTE_COLL_H

// Get the base class and various definitions.
#include "u/nvasil/tpie/bte_coll_base.h"

// The MMAP implementation.
#include "u/nvasil/tpie/bte_coll_mmap.h"

// The UFS implementation.
#include "u/nvasil/tpie/bte_coll_ufs.h"

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"


#if defined(BTE_COLLECTION_IMP_MMB)
//	TPIE_OS_UNIX_ONLY_WARNING_BTE_COLLECTION_IMP_MMB_UNIX_ONLY
#  define BTE_COLLECTION_IMP_MMAP
#endif

#define _BTE_COLL_IMP_COUNT (defined(BTE_COLLECTION_IMP_UFS) + \
                             defined(BTE_COLLECTION_IMP_MMAP) + \
                             defined(BTE_COLLECTION_IMP_USER_DEFINED))

// Multiple implem. are included, but we have to choose a default one.
#if (_BTE_COLL_IMP_COUNT > 1)
//	TPIE_OS_UNIX_ONLY_WARNING_MULTIPLE_BTE_COLLECTION_IMP_DEFINED
#  define BTE_COLLECTION_IMP_MMAP
#elif (_BTE_COLL_IMP_COUNT == 0)
//	TPIE_OS_UNIX_ONLY_WARNING_NO_DEFAULT_BTE_COLLECTION
#  define BTE_COLLECTION_IMP_MMAP
#endif

#define BTE_COLLECTION_MMAP BTE_collection_mmap<TPIE_BLOCK_ID_TYPE>
#define BTE_COLLECTION_UFS  BTE_collection_ufs<TPIE_BLOCK_ID_TYPE>

#if defined(BTE_COLLECTION_IMP_MMAP)
#  define BTE_COLLECTION BTE_COLLECTION_MMAP
#elif defined(BTE_COLLECTION_IMP_UFS)
#  define BTE_COLLECTION BTE_COLLECTION_UFS
#elif defined(BTE_COLLECTION_IMP_USER_DEFINED)
   // Do not define BTE_COLLECTION. The user will define it.
#endif

#endif // _BTE_COLL_H
