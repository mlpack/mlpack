//
// File:    bte_coll_mmap.h (formerly bte_coll_mmb.h)
// Author:  Octavian Procopiuc <tavi@cs.duke.edu>
//
// $Id: bte_coll_mmap.h,v 1.12 2005/01/14 18:58:32 tavi Exp $
//
// BTE_collection_mmap class definition.
//
#ifndef _BTE_COLL_MMAP_H
#define _BTE_COLL_MMAP_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"
// Get the base class.
#include "u/nvasil/tpie/bte_coll_base.h"

// For header's type field (77 == 'M').
#define BTE_COLLECTION_MMAP_ID 77

// Define write behavior, if not already defined by the user.
// Allowed values:
//  0    (synchronous writes)
//  1    (asynchronous writes using MS_ASYNC - see msync(2))
//  2    (asynchronous bulk writes) [default]
#ifndef BTE_COLLECTION_MMAP_LAZY_WRITE 
#  define BTE_COLLECTION_MMAP_LAZY_WRITE 2
#endif

template<class BIDT = TPIE_BLOCK_ID_TYPE>
class BTE_collection_mmap: public BTE_collection_base<BIDT> {
  protected:
  using BTE_collection_base<BIDT>::header_;
  using BTE_collection_base<BIDT>::freeblock_stack_;
  using BTE_collection_base<BIDT>::bcc_fd_;
  using BTE_collection_base<BIDT>::per_;
  using BTE_collection_base<BIDT>::os_block_size_;
  using BTE_collection_base<BIDT>::base_file_name_;
  using BTE_collection_base<BIDT>::status_;
  using BTE_collection_base<BIDT>::read_only_;
  using BTE_collection_base<BIDT>::in_memory_blocks_;
  using BTE_collection_base<BIDT>::file_pointer;
  using BTE_collection_base<BIDT>::stats_;
  using BTE_collection_base<BIDT>::gstats_;
  using BTE_collection_base<BIDT>::register_memory_allocation;
  using BTE_collection_base<BIDT>::register_memory_deallocation;
  using BTE_collection_base<BIDT>::bid_to_file_offset;
  using BTE_collection_base<BIDT>::create_stack;
  using BTE_collection_base<BIDT>::new_block_getid;
  using BTE_collection_base<BIDT>::delete_block_shared;
  

  public:
  // Constructor. Read and verify the header of the
  // collection. Implemented in the base class.
  BTE_collection_mmap(const char *base_file_name,
		     BTE_collection_type type = BTE_WRITE_COLLECTION,
		     size_t logical_block_factor = 1):
    BTE_collection_base<BIDT>(base_file_name, type, logical_block_factor, TPIE_OS_FLAG_USE_MAPPING_TRUE) {
    header_.type = BTE_COLLECTION_MMAP_ID;
  }

  // Allocate a new block in block collection and then map that block
  // into memory, allocating and returning an appropriately
  // initialized Block. Main memory usage increases.
  BTE_err new_block(BIDT &bid, void * &place) {
    BTE_err err;
    // Get a block id.
    if ((err = new_block_getid(bid)) != BTE_ERROR_NO_ERROR)
      return err;
    // We have a bid, so we can call the get_block routine.
    if ((err = get_block_internals(bid, place)) != BTE_ERROR_NO_ERROR)
      return err;   
    header_.used_blocks++;
    stats_.record(BLOCK_NEW);
    gstats_.record(BLOCK_NEW);
    return BTE_ERROR_NO_ERROR;
  }

  // Delete a previously created, currently mapped-in BLOCK. This causes the
  // number of free blocks in the collection to increase by 1, the bid is
  // entered into the stdio_stack.  NOTE that it is the onus of the user of
  // this class to ensure that the bid of this placeholder is correct. No
  // check is made if the bid is an invalid or previously unallocated bid,
  // which will introduce erroneous entries in the stdio_stack of free
  // blocks. Main memory usage goes down.
  BTE_err delete_block(BIDT bid, void * place) {
    BTE_err err;
    if ((err = put_block_internals(bid, place, 1)) != BTE_ERROR_NO_ERROR)  
      return err; 
    if ((err = delete_block_shared(bid)) != BTE_ERROR_NO_ERROR)
      return err;
    header_.used_blocks--;
    stats_.record(BLOCK_DELETE);
    gstats_.record(BLOCK_DELETE);
    return BTE_ERROR_NO_ERROR;
  }

  // Map in the block with the indicated bid and allocate and initialize a
  // corresponding placeholder. NOTE once more that it is the user's onus
  // to ensure that the bid requested corresponds to a valid block and so
  // on; no checks made here to ensure that that is indeed the case. Main
  // memory usage increases.
  BTE_err get_block(BIDT bid, void * &place) {
    BTE_err err;
    if ((err = get_block_internals(bid, place)) != BTE_ERROR_NO_ERROR)
      return err;
    stats_.record(BLOCK_GET);
    gstats_.record(BLOCK_GET);
    return BTE_ERROR_NO_ERROR;
  }

  // Unmap a currently mapped in block. NOTE once more that it is the user's
  // onus to ensure that the bid is correct and so on; no checks made here
  // to ensure that that is indeed the case. Main memory usage decreases.
  BTE_err put_block(BIDT bid, void * place, char dirty = 1) {
    BTE_err err;
    if ((err = put_block_internals(bid, place, dirty)) != BTE_ERROR_NO_ERROR)
      return err;
    stats_.record(BLOCK_PUT);
    gstats_.record(BLOCK_PUT);
    return BTE_ERROR_NO_ERROR;
  }

  // Synchronize the in-memory block with the on-disk block.
  BTE_err sync_block(BIDT bid, void* place, char dirty = 1);

protected:
  BTE_err get_block_internals(BIDT bid, void *&place);
  BTE_err put_block_internals(BIDT bid, void* place, char dirty);
};


template<class BIDT>
BTE_err BTE_collection_mmap<BIDT>::get_block_internals(BIDT bid, void * &place) {

  place = TPIE_OS_MMAP(NULL, header_.block_size,
	       read_only_ ? TPIE_OS_FLAG_PROT_READ : 
	       TPIE_OS_FLAG_PROT_READ | TPIE_OS_FLAG_PROT_WRITE, 
#ifdef SYSTYPE_BSD
	       MAP_FILE | MAP_VARIABLE | MAP_NOSYNC |
#endif
	       TPIE_OS_FLAG_MAP_SHARED, bcc_fd_, bid_to_file_offset(bid));

  if (place == (void *)(-1)) {
   TP_LOG_FATAL_ID("mmap() failed to map in a block from file.");
   TP_LOG_FATAL_ID(strerror(errno));
    return BTE_ERROR_MEMORY_ERROR;
  }

  //  madvise(place, header_.block_size, MADV_RANDOM);

  // Register the memory allocation since mmapped memory is
  // not accounted for otherwise.
  register_memory_allocation(header_.block_size);

  in_memory_blocks_++;
  return BTE_ERROR_NO_ERROR;
}


template<class BIDT>
BTE_err BTE_collection_mmap<BIDT>::put_block_internals(BIDT bid, void* place, char dirty) {
  
  // The dirty parameter is not used in this implemetation.

  if ((bid <= 0) || (bid >= header_.last_block)) {
   TP_LOG_FATAL_ID("Incorrect bid in placeholder.");
    return BTE_ERROR_INVALID_PLACEHOLDER;
  }

#if (BTE_COLLECTION_MMAP_LAZY_WRITE < 2)
  if (!read_only_) {
    if (TPIE_OS_MSYNC((char*)place, header_.block_size, 
#  if (BTE_COLLECTION_MMAP_LAZY_WRITE == 1)
	      TPIE_OS_FLAG_MS_ASYNC
#  else
	      TPIE_OS_FLAG_MS_SYNC
#  endif
	      ) == -1) {
     TP_LOG_FATAL_ID("Failed to msync() block to file.");
     TP_LOG_FATAL_ID(strerror(errno));
      return BTE_ERROR_IO_ERROR;
    }    
  }
#endif

  if (TPIE_OS_MUNMAP((char*)place, header_.block_size) == -1) {
   TP_LOG_FATAL_ID("Failed to unmap() block of file.");
   TP_LOG_FATAL_ID(strerror(errno));
    return BTE_ERROR_IO_ERROR;
  }

  register_memory_deallocation(header_.block_size);

  in_memory_blocks_--;
  return BTE_ERROR_NO_ERROR;
}
 

template<class BIDT>
BTE_err BTE_collection_mmap<BIDT>::sync_block(BIDT bid, void* place, char dirty) {

  if ((bid <= 0) || (bid >= header_.last_block)) {
   TP_LOG_FATAL_ID("Incorrect bid in placeholder.");
    return BTE_ERROR_INVALID_PLACEHOLDER;
  }
  
  if (!read_only_) {
    if (TPIE_OS_MSYNC((char*)place, header_.block_size, TPIE_OS_FLAG_MS_SYNC)) {
     TP_LOG_FATAL_ID("Failed to msync() block to file.");
     TP_LOG_FATAL_ID(strerror(errno));
      return BTE_ERROR_IO_ERROR;
    }
  }

  stats_.record(BLOCK_SYNC);
  gstats_.record(BLOCK_SYNC);
  return BTE_ERROR_NO_ERROR;
}

#endif //_BTE_COLL_MMAP_H
