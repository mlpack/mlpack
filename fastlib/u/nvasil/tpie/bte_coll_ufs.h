//
// File:    bte_coll_ufs.h
// Author:  Octavian Procopiuc <tavi@cs.duke.edu>
//
// $Id: bte_coll_ufs.h,v 1.12 2005/01/14 18:58:32 tavi Exp $
//
// BTE_collection_ufs class definition.
//
#ifndef _BTE_COLL_UFS_H
#define _BTE_COLL_UFS_H

// Get definitions for working with Unix and Windows
#include <portability.h>
// Get the base class.
#include <bte_coll_base.h>

// For header's type field (85 == 'U').
#define BTE_COLLECTION_UFS_ID 85

template<class BIDT = TPIE_BLOCK_ID_TYPE>
class BTE_collection_ufs: public BTE_collection_base<BIDT> {
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

  // Constructors.
  BTE_collection_ufs(const char *base_file_name,
		     BTE_collection_type type = BTE_WRITE_COLLECTION,
		     size_t logical_block_factor = 1):
    BTE_collection_base<BIDT>(base_file_name, type, logical_block_factor) {
    header_.type = BTE_COLLECTION_UFS_ID;
  }

  // Allocate a new block in block collection and then read that block into
  // memory, allocating and returning an appropriately initialized
  // Block. Main memory usage increases.
  BTE_err new_block(BIDT &bid, void * &place) {
    BTE_err err;
    // Get a block id.
    if ((err = new_block_getid(bid)) != BTE_ERROR_NO_ERROR)
      return err;
    // We have a bid, so we can call the get_block routine.
    if ((err = new_block_internals(bid, place)) != BTE_ERROR_NO_ERROR)
      return err;
    header_.used_blocks++;
    stats_.record(BLOCK_NEW);
    gstats_.record(BLOCK_NEW);
    return BTE_ERROR_NO_ERROR;
  }

  // Delete a previously created, currently in-memory BLOCK. This causes
  // the number of free blocks in the collection to increase by 1, the bid
  // is entered into the stdio_stack.  NOTE that it is the onus of the user
  // of this class to ensure that the bid of this placeholder is
  // correct. No check is made if the bid is an invalid or previously
  // unallocated bid, which will introduce erroneous entries in the
  // stdio_stack of free blocks. Main memory usage goes down.
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

  // Read the block with the indicated bid and allocate and initialize a
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

  // Write a currently in-memory block. NOTE once more that it is the
  // user's onus to ensure that the bid is correct and so on; no checks
  // made here to ensure that that is indeed the case. Main memory usage
  // decreases.
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

  BTE_err new_block_internals(BIDT bid, void *&place);
  //  BTE_err new_block_getid_specific(BIDT& bid);
  BTE_err get_block_internals(BIDT bid, void *&place);
  BTE_err put_block_internals(BIDT bid, void* place, char dirty);
};

template<class BIDT>
BTE_err BTE_collection_ufs<BIDT>::new_block_internals(BIDT bid, void* &place) {
  if ((place = new char[header_.block_size]) == NULL) {    
   TP_LOG_FATAL_ID("new() failed to alloc space for a block from file.");
    return BTE_ERROR_MEMORY_ERROR;
  }

  in_memory_blocks_++;
  return BTE_ERROR_NO_ERROR;
}

#if 0
template<class BIDT>
BTE_err BTE_collection_ufs<BIDT>::new_block_getid_specific(BIDT& bid) {
    BIDT *lbn;
    BTE_err err;
    if (header_.used_blocks < header_.last_block - 1) {

      tp_assert(freeblock_stack_ != NULL, 
		"BTE_collection_ufs internal error: NULL stack pointer");

      size_t slen = freeblock_stack_->stream_len();
      tp_assert(slen > 0, "BTE_collection_ufs internal error: empty stack");
      if ((err = freeblock_stack_->pop(&lbn)) != BTE_ERROR_NO_ERROR)
	return err;
      bid = *lbn;

    } else {

      tp_assert(header_.last_block <= header_.total_blocks, 
		"BTE_collection_ufs internal error: last_block>total_blocks");

      if (header_.last_block == header_.total_blocks) {
	// Increase the capacity for storing blocks in the stream by
	// 16 (only by 1 the first time around to be gentle with very
	// small coll's).
	if (header_.total_blocks == 1)
	  header_.total_blocks += 2;
	else if (header_.total_blocks <= 161)
	  header_.total_blocks += 8;
	else
	  header_.total_blocks += 64;
#define USE_FTRUNCATE_FOR_UFS 1
#if     USE_FTRUNCATE_FOR_UFS
	if (TPIE_OS_FTRUNCATE(bcc_fd_, bid_to_file_offset(header_.total_blocks))) {
	 TP_LOG_FATAL_ID("Failed to ftruncate() to the new end of file.");
	 TP_LOG_FATAL_ID(strerror(errno));
	  return BTE_ERROR_OS_ERROR;
	}
#else
	TPIE_OS_OFFSET curr_off;
	char* tbuf = new char[header_.os_block_size];
	if ((curr_off = TPIE_OS_LSEEK(bcc_fd_, 0, TPIE_OS_FLAG_SEEK_END)) == (TPIE_OS_OFFSET)-1) {
	 TP_LOG_FATAL_ID("Failed to lseek() to the end of file.");
	 TP_LOG_FATAL_ID(strerror(errno));
	  return BTE_ERROR_OS_ERROR;
	}
	while (curr_off < bid_to_file_offset(header_.total_blocks)) {
	  TPIE_OS_WRITE(bcc_fd_, tbuf, header_.os_block_size);
	  curr_off += header_.os_block_size;
	}
	file_pointer = curr_off;
	delete [] tbuf;
#endif
      }
      bid = header_.last_block++;
    }
    return BTE_ERROR_NO_ERROR;
}
#endif

template<class BIDT>
BTE_err BTE_collection_ufs<BIDT>::get_block_internals(BIDT bid, void * &place) {

  if ((place = new char[header_.block_size]) == NULL) {    
   TP_LOG_FATAL_ID("new() failed to alloc space for a block from file.");
    return BTE_ERROR_MEMORY_ERROR;
  }

  if (file_pointer != bid_to_file_offset(bid)) {
    if (TPIE_OS_LSEEK(bcc_fd_, bid_to_file_offset(bid), TPIE_OS_FLAG_SEEK_SET) !=  
	bid_to_file_offset(bid)) {
     TP_LOG_FATAL_ID("lseek failed in file.");
      return BTE_ERROR_IO_ERROR;
    }
  }

  if (TPIE_OS_READ(bcc_fd_, (char *) place, header_.block_size) != 
      (TPIE_OS_SSIZE_T)header_.block_size) {
   TP_LOG_FATAL_ID("Failed to read() from file.");
    return BTE_ERROR_IO_ERROR;
  }

  file_pointer = bid_to_file_offset(bid) + header_.block_size;

  in_memory_blocks_++;
  return BTE_ERROR_NO_ERROR;
}


template<class BIDT>
BTE_err BTE_collection_ufs<BIDT>::put_block_internals(BIDT bid, void * place, char dirty) {
  
  if ((bid < 0) || (bid >= header_.last_block)) {
   TP_LOG_FATAL_ID("Incorrect bid in placeholder.");
    return BTE_ERROR_INVALID_PLACEHOLDER;
  }
  
  if (place == NULL) {
   TP_LOG_FATAL_ID("Null block ptr field in placeholder.");
    return BTE_ERROR_INVALID_PLACEHOLDER;
  }

  //  if (place->dirty) 

  if (!read_only_) {

    if (file_pointer != bid_to_file_offset(bid)) {
      if (TPIE_OS_LSEEK(bcc_fd_, bid_to_file_offset(bid), TPIE_OS_FLAG_SEEK_SET) 
	  !=  bid_to_file_offset(bid)) {
	TP_LOG_FATAL_ID("Failed to lseek() in file.");
	return BTE_ERROR_IO_ERROR;
      }
    }

    if (TPIE_OS_WRITE(bcc_fd_, place, header_.block_size) != (TPIE_OS_SSIZE_T)header_.block_size) {
     TP_LOG_FATAL_ID("Failed to write() block to file.");
      return BTE_ERROR_IO_ERROR;
    }

    file_pointer = bid_to_file_offset(bid) + header_.block_size;
  }

  delete [] (char *) place;

  in_memory_blocks_--;
  return BTE_ERROR_NO_ERROR;
}

template<class BIDT>
BTE_err BTE_collection_ufs<BIDT>::sync_block(BIDT bid, void* place, char dirty) {

  if ((bid < 0) || (bid >= header_.last_block)) {
   TP_LOG_FATAL_ID("Incorrect bid in placeholder.");
    return BTE_ERROR_INVALID_PLACEHOLDER;
  }
  
  if (place == NULL) {
   TP_LOG_FATAL_ID("Null block pointer.");
    return BTE_ERROR_INVALID_PLACEHOLDER;
  }

  if (!read_only_) {
    
    if (TPIE_OS_LSEEK(bcc_fd_, bid_to_file_offset(bid), TPIE_OS_FLAG_SEEK_SET) 
	!=  bid_to_file_offset(bid)) {
     TP_LOG_FATAL_ID("Failed to lseek() in file.");
      return BTE_ERROR_IO_ERROR;
    }
    
    if (TPIE_OS_WRITE(bcc_fd_, place, header_.block_size) != (TPIE_OS_SSIZE_T)header_.block_size) {
     TP_LOG_FATAL_ID("Failed to write() block to file.");
      return BTE_ERROR_IO_ERROR;
    }

    file_pointer = bid_to_file_offset(bid) + header_.block_size;
  }
 
  stats_.record(BLOCK_SYNC);
  gstats_.record(BLOCK_SYNC);
  return BTE_ERROR_NO_ERROR;
}

#endif // _BTE_COLL_UFS_H
