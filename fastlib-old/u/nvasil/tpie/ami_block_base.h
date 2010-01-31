// Copyright (C) 2001 Octavian Procopiuc
//
// File:    ami_block_base.h
// Author:  Octavian Procopiuc <tavi@cs.duke.edu>
//
// $Id: ami_block_base.h,v 1.14 2004/08/17 16:47:38 jan Exp $
//
// Definition of the AMI_block_base class and supporting types:
// AMI_bid, AMI_block_status.
//

#ifndef _AMI_BLOCK_BASE_H
#define _AMI_BLOCK_BASE_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

// The AMI error codes.
#include "u/nvasil/tpie/ami_err.h"
// The AMI_COLLECTION class.
#include "u/nvasil/tpie/ami_coll.h"

// AMI block id type.
typedef TPIE_BLOCK_ID_TYPE AMI_bid;

// Block status type.
enum AMI_block_status {
  AMI_BLOCK_STATUS_VALID = 0,
  AMI_BLOCK_STATUS_INVALID = 1
};


template<class BTECOLL>
class AMI_block_base {
public:
  //  typedef typename BTECOLL::block_id_t id_t;

protected:

  // Pointer to the block collection.
  BTECOLL * pcoll_;

  // Unique ID. Represents the offset of the block in the blocks file.
  AMI_bid bid_;

  // Dirty bit. If set, the block needs to be written back.
  char dirty_;

  // Pointer to the actual data.
  void * pdata_;

  // Persistence flag.
  persistence per_;

public:

  // Constructor.
  // Read and initialize a block with a given ID.
  // When bid is missing or 0, a new block is created.
  AMI_block_base(AMI_collection_single<BTECOLL>* pacoll, AMI_bid bid = 0)
    : bid_(bid), dirty_(0), per_(PERSIST_PERSISTENT) {
    pcoll_ = pacoll->bte();
    if (bid != 0) {
      // Get an existing block from disk.
      if (pcoll_->get_block(bid_, pdata_) != BTE_ERROR_NO_ERROR)
	pdata_ = NULL;
    } else {
      // Create a new block in the collection.
      if (pcoll_->new_block(bid_, pdata_) != BTE_ERROR_NO_ERROR)
	pdata_ = NULL;
    }
  }

  AMI_err sync() {
    if (pcoll_->sync_block(bid_, pdata_) != BTE_ERROR_NO_ERROR)
      return AMI_ERROR_BTE_ERROR;
    else
      return AMI_ERROR_NO_ERROR;
  }

  // Get the block id.
  AMI_bid bid() const { return bid_; }

  // Get a reference to the dirty bit.
  char& dirty() { return dirty_; };
  char dirty() const { return dirty_; }

  // Copy block rhs into this block.
  AMI_block_base<BTECOLL>& operator=(const AMI_block_base<BTECOLL>& rhs) { 
    if (pcoll_ == rhs.pcoll_) {
      memcpy(pdata_, rhs.pdata_, pcoll_->block_size());
      dirty_ = 1;
    } else 
      pdata_ = NULL;
    return *this; 
  }

  // Get the block's status.
  AMI_block_status status() const { 
    return (pdata_ == NULL) ? 
      AMI_BLOCK_STATUS_INVALID: AMI_BLOCK_STATUS_VALID; 
  }

  // Return true if the block is valid.
  bool is_valid() const {
    return (pdata_ != NULL);
  }

  // Return true if the block is invalid.
  bool operator!() const {
    return (pdata_ == NULL);
  }

  void persist(persistence per) { per_ = per; }

  persistence persist() const { return per_; }

  size_t block_size() const { return pcoll_->block_size(); }

  // Destructor.
  ~AMI_block_base() {
    // Check first the status of the collection. 
    if (pdata_ != NULL){
      if (per_ == PERSIST_PERSISTENT) {
        // Write back the block.
        pcoll_->put_block(bid_, pdata_); 
      } else {
	// Delete the block from the collection.
	pcoll_->delete_block(bid_, pdata_);
      }
    }
  }
};


#endif //_AMI_BLOCK_BASE_H
