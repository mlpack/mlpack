// Copyright (c) 2001 Octavian Procopiuc
//
// File:    bte_coll_base.h
// Authors: Octavian Procopiuc <tavi@cs.duke.edu>
//          (using some code by Rakesh Barve)
//
// $Id: bte_coll_base.h,v 1.27 2004/08/17 16:48:06 jan Exp $
//
// BTE_collection_base class and various basic definitions.


#ifndef _BTE_COLL_BASE_H
#define _BTE_COLL_BASE_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

// Include the registration based memory manager.
#define MM_IMP_REGISTER
#include "u/nvasil/tpie/mm.h"

// For persist.
#include "u/nvasil/tpie/persist.h"
// For BTE_stack_ufs
#include "u/nvasil/tpie/bte_stack_ufs.h"
// For BTE_err.
#include "u/nvasil/tpie/bte_err.h"
// For class tpie_stats_collection.
#include "u/nvasil/tpie/tpie_stats_coll.h"

// BTE_COLLECTION types passed to constructors.
enum BTE_collection_type {
    BTE_READ_COLLECTION = 1,    // Open existing stream read only.
    BTE_WRITE_COLLECTION,	// Open for read/write. Create if non-existent.
    BTE_WRITE_NEW_COLLECTION    // Open for read/write a new collection,
    // even if a nonempty file with that name exists.
};

// BTE collection status.
enum BTE_collection_status {
    BTE_COLLECTION_STATUS_VALID = 0,
    BTE_COLLECTION_STATUS_INVALID = 1
};


// Maximum length of the file names.
#define BTE_COLLECTION_PATH_NAME_LEN 128

// Number of bytes in the header's user_data_ field.
#define BTE_COLLECTION_USER_DATA_LEN 512

// The magic number of the files storing blocks.
// (in network byteorder, it spells "TPBC": TPie Block Collection)
#define BTE_COLLECTION_HEADER_MAGIC_NUMBER 0x54504243

// Default file name suffixes
#define BTE_COLLECTION_BLK_SUFFIX ".blk"
#define BTE_COLLECTION_STK_SUFFIX ".stk"

// Setting this to 1 causes the use of ftruncate(2) for extending
// files, which, in conjunction with mmap(2), results in more
// fragmented files and, consequently, slower I/O. See mmap(2) on
// FreeBSD for an explanation. When set to 0, lseek(2) and write(2)
// are used to extend the files. This should be set to 1 for WIN32
// (see portability.h)
#ifndef BTE_COLLECTION_USE_FTRUNCATE
#define BTE_COLLECTION_USE_FTRUNCATE 0
#endif


// The in-memory representation of the BTE_COLLECTION header.
// This data structure is read from/written to the first 
// (physical) page of the blocks file.

class BTE_collection_header {
public:

    // Unique header identifier. Set to BTE_COLLECTION_HEADER_MAGIC_NUMBER
    unsigned int magic_number;
    // Should be 1 for current version.
    unsigned int version;
    // The type of BTE_COLLECTION that created this header. Setting this
    // field is optional and is mostly for information purposes and
    // similarity with stream header. The current implementations all
    // use the same file format; it's not important to differentiate
    // among them, since they can all read each other's collections. If
    // used, it should be set to a non-zero value (zero is reserved for
    // the base class).
    unsigned int type;
    // The number of bytes in this structure.
    TPIE_OS_SIZE_T header_length;
    // The number of blocks consumed by this collection, plus 1.
    TPIE_OS_OFFSET total_blocks;
    // The highest bid any block of this block collection has, PLUS 1
    // (always <= total_blocks).
    TPIE_OS_OFFSET last_block; 
    // The number of valid blocks in this block collection.
    TPIE_OS_OFFSET used_blocks;
    // The size of a physical block on the device this stream resides.
    TPIE_OS_SIZE_T os_block_size;
    // Size in bytes of each logical block.
    TPIE_OS_SIZE_T block_size;
    // Some data to be filled by the user of the collection.
    char user_data[BTE_COLLECTION_USER_DATA_LEN];
  
    // Default constructor.
  BTE_collection_header():
    magic_number(BTE_COLLECTION_HEADER_MAGIC_NUMBER), 
      version(1), 
      type(0),
      header_length(sizeof(BTE_collection_header)), 
      total_blocks(1), 
      last_block(1), 
      used_blocks(0) {
      os_block_size = TPIE_OS_BLOCKSIZE();
  }
};

// A base class for all implementations of block collection classes.
template <class BIDT>
class BTE_collection_base {
protected:

    // Various parameters (will be stored into the file header block).
    BTE_collection_header header_;

    // A stack of TPIE_OS_OFFSET's.
    BTE_stack_ufs<BIDT> *freeblock_stack_; 

    // File descriptor for the file backing the block collection.
    TPIE_OS_FILE_DESCRIPTOR bcc_fd_;

    char base_file_name_[BTE_COLLECTION_PATH_NAME_LEN];

    TPIE_OS_SIZE_T os_block_size_;

    // Persistency flag. Set during construction and using the persist()
    // method.
    persistence per_;

    // Status of the collection. Set during construction.
    BTE_collection_status status_;

    // Read-only flag. Set during construction.
    bool read_only_;

    // Number of blocks from this collection that are currently in memory
    TPIE_OS_SIZE_T in_memory_blocks_;

    // File pointer position. A value of -1 signals unknown position.
    TPIE_OS_OFFSET file_pointer;

    // Statistics for this object.
    tpie_stats_collection stats_;

    // Global collection statistics.
    static tpie_stats_collection gstats_;

private:
    // Helper functions. We don't want them inherited.

    // Initialization common to all constructors.
    void shared_init(BTE_collection_type type, size_t logical_block_factor, TPIE_OS_MAPPING_FLAG mapping);

    // Read header from disk.
    BTE_err read_header(char *bcc_name);

    // Write header to disk.
    BTE_err write_header(char* bcc_name);

    void remove_stack_file();

protected:

    // Needs to be inlined!
    BTE_err register_memory_allocation(TPIE_OS_SIZE_T sz) {
	if (MM_manager.register_allocation(sz) != MM_ERROR_NO_ERROR) {
	    status_ = BTE_COLLECTION_STATUS_INVALID;
	   TP_LOG_FATAL_ID("Memory manager error in allocation.");
	    return BTE_ERROR_MEMORY_ERROR;
	}
	return BTE_ERROR_NO_ERROR;
    }

    // Needs to be inlined!
    BTE_err register_memory_deallocation(TPIE_OS_SIZE_T sz) {
	if (MM_manager.register_deallocation(sz) != MM_ERROR_NO_ERROR) {
	    status_ = BTE_COLLECTION_STATUS_INVALID;
	   TP_LOG_FATAL_ID("Memory manager error in deallocation.");
	    return BTE_ERROR_MEMORY_ERROR;
	}
	return BTE_ERROR_NO_ERROR;
    }

    TPIE_OS_OFFSET bid_to_file_offset(BIDT bid) const 
    { return header_.os_block_size + header_.block_size * (bid-1); }

    void create_stack();

    // Common code for all new_block implementations. Inlined.
    BTE_err new_block_getid(BIDT& bid) {
	// We try getting a free bid from the stack first. If there aren't
	// any there, we will try to get one after last_block; if there are
	// no blocks past last_block, we will ftruncate() some more blocks
	// to the tail of the BCC and then get a free bid.
	BIDT *lbn;
	BTE_err err;
	if (header_.used_blocks < header_.last_block - 1) {
	    tp_assert(freeblock_stack_ != NULL, 
		      "BTE_collection_ufs internal error: NULL stack pointer");
	    // TODO: this is a costly operation. improve!
	    TPIE_OS_OFFSET slen = freeblock_stack_->stream_len();
	    tp_assert(slen > 0, "BTE_collection_ufs internal error: empty stack");
	    if ((err = freeblock_stack_->pop(&lbn)) != BTE_ERROR_NO_ERROR)
		return err;
	    bid = *lbn;
	} else {
	    tp_assert(header_.last_block <= header_.total_blocks, 
		      "BTE_collection_ufs internal error: last_block>total_blocks");
	    if (header_.last_block == header_.total_blocks) {
		// Increase the capacity for storing blocks in the stream by
		// 16 (only by 2 the first time around to be gentle with very
		// small coll's).
		if (header_.total_blocks == 1)
		    header_.total_blocks += 2;
		else if (header_.total_blocks <= 161)
		    header_.total_blocks += 8;
		else
		    header_.total_blocks += 64;


#if BTE_COLLECTION_USE_FTRUNCATE
		if (TPIE_OS_FTRUNCATE(bcc_fd_, bid_to_file_offset(header_.total_blocks))) {
		   TP_LOG_FATAL_ID("Failed to truncate to the new end of file.");
		    //LOG_FATAL_ID(strerror(errno));
		    return BTE_ERROR_OS_ERROR;
		}
#else
		char* tbuf = new char[header_.os_block_size];
		TPIE_OS_OFFSET curr_off;

		if ((curr_off = TPIE_OS_LSEEK(bcc_fd_, 0, TPIE_OS_FLAG_SEEK_END)) == (TPIE_OS_OFFSET)(-1)) {
		   TP_LOG_FATAL_ID("Failed to seek to the end of file.");
		    //LOG_FATAL_ID(strerror(errno));
		    return BTE_ERROR_OS_ERROR;
		}
		while (curr_off < bid_to_file_offset(header_.total_blocks)) {
		    TPIE_OS_WRITE(bcc_fd_, tbuf, header_.os_block_size);
		    curr_off += header_.os_block_size;
		}
		delete [] tbuf;
		file_pointer = curr_off;
#endif

	    }
	    bid = header_.last_block++;
	}
	return BTE_ERROR_NO_ERROR;
    }

    // Common code for all delete_block implementations. Inlined.
    BTE_err delete_block_shared(BIDT bid) {
	if (bid == header_.last_block - 1) 
	    header_.last_block--;
	else {
	    if (freeblock_stack_ == NULL)
		create_stack();
	    //tp_assert(freeblock_stack_ != NULL, 
	    //	"BTE_collection_ufs internal error: NULL stack pointer");
	    return freeblock_stack_->push(bid);
	}
	return BTE_ERROR_NO_ERROR;
    }

public:

    typedef BIDT block_id_t;

    BTE_collection_base(const char *base_name, BTE_collection_type ct, 
			size_t logical_block_factor, TPIE_OS_MAPPING_FLAG mapping = TPIE_OS_FLAG_USE_MAPPING_FALSE);

    // Return the total number of used blocks.
	TPIE_OS_OFFSET size() const { return header_.used_blocks; }
                          	
    // Return the total number of blocks consumed by the block collection.
	TPIE_OS_OFFSET file_size() const { return header_.total_blocks - 1; }

    // Return the logical block size in bytes.
	TPIE_OS_SIZE_T block_size() const { return header_.block_size; }

    // Return the logical block factor.
	TPIE_OS_SIZE_T block_factor() const 
	{ return header_.block_size / header_.os_block_size; }

    // Return the status of the collection.
    BTE_collection_status status() const { return status_; }

    // Set the persistence flag. 
    void persist(persistence p) { per_ = p; }

    // Inquire the persistence status.
    persistence persist() const { return per_; }

    const char *base_file_name() const { return base_file_name_; }

    void *user_data() { return (void *) header_.user_data; }

    // Local statistics (for this object).
    const tpie_stats_collection& stats() const { return stats_; }

    // Global statistics (for all collections).
    static const tpie_stats_collection& gstats() { return gstats_; }

    // Destructor.
    ~BTE_collection_base(); 

#if defined(__sun__) 
    static bool direct_io;
#endif
};


template<class BIDT>
tpie_stats_collection BTE_collection_base<BIDT>::gstats_;

template<class BIDT>
void BTE_collection_base<BIDT>::create_stack() {
  // Fill in the stack file name.
  char stack_name[BTE_COLLECTION_PATH_NAME_LEN];
  strncpy((char *) stack_name, base_file_name_, BTE_COLLECTION_PATH_NAME_LEN - 4);
  strcat((char *) stack_name, BTE_COLLECTION_STK_SUFFIX);

  // Construct the pre-existing freeblock_stack.
  freeblock_stack_ = new BTE_stack_ufs<BIDT>((char *) stack_name, 
		       read_only_? BTE_READ_STREAM: BTE_WRITE_STREAM);
  
}

template<class BIDT>
void BTE_collection_base<BIDT>::remove_stack_file() {
   // Fill in the stack file name.
  char stack_name[BTE_COLLECTION_PATH_NAME_LEN];
  strncpy((char *) stack_name, base_file_name_, BTE_COLLECTION_PATH_NAME_LEN - 4);
  strcat((char *) stack_name, BTE_COLLECTION_STK_SUFFIX);

  TPIE_OS_UNLINK(stack_name);
}

template<class BIDT>
BTE_collection_base<BIDT>::BTE_collection_base(const char *base_name, 
		 BTE_collection_type type, size_t logical_block_factor, TPIE_OS_MAPPING_FLAG mapping):
  header_(), freeblock_stack_(NULL) {

  if (base_name == NULL) {
    status_ = BTE_COLLECTION_STATUS_INVALID;
   TP_LOG_FATAL_ID("NULL file name passed to constructor");
    return;
  }
  
  strncpy((char*) base_file_name_, base_name, BTE_COLLECTION_PATH_NAME_LEN - 4);

  // A collection with a given name is not deleted upon destruction.
  per_ = PERSIST_PERSISTENT;

  shared_init(type, logical_block_factor, mapping);
}


template<class BIDT>
void BTE_collection_base<BIDT>::shared_init(BTE_collection_type type,
				      TPIE_OS_SIZE_T logical_block_factor, TPIE_OS_MAPPING_FLAG mapping) {
  read_only_ = (type == BTE_READ_COLLECTION);
  status_ = BTE_COLLECTION_STATUS_VALID;
  in_memory_blocks_ = 0;
  file_pointer = -1;
  os_block_size_ = TPIE_OS_BLOCKSIZE();

  // Fill in the blocks file name.
  char bcc_name[BTE_COLLECTION_PATH_NAME_LEN];
  strncpy((char *) bcc_name, base_file_name_, BTE_COLLECTION_PATH_NAME_LEN - 4);  
  strcat((char *) bcc_name, BTE_COLLECTION_BLK_SUFFIX);

  if (read_only_) {

    if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(bcc_fd_ = TPIE_OS_OPEN_ORDONLY(bcc_name, mapping))) {
      status_ = BTE_COLLECTION_STATUS_INVALID;
     TP_LOG_FATAL_ID("open() failed to open read-only file: ");
     TP_LOG_FATAL_ID(bcc_name);	
      return;
    }

    if (read_header(bcc_name) != BTE_ERROR_NO_ERROR) {
      status_ = BTE_COLLECTION_STATUS_INVALID;
      return;
    }

    // Check whether we need a stack.
    if (header_.used_blocks < header_.last_block - 1) {
      create_stack();
      if (freeblock_stack_->status() == BTE_STREAM_STATUS_INVALID) {
	status_ = BTE_COLLECTION_STATUS_INVALID;
	return;
      }
    } else
      freeblock_stack_ = NULL;

  } else  {   // Writeable bcc.

    // If a new collection, remove any existing files with the same names.
    if (type == BTE_WRITE_NEW_COLLECTION) {
      TPIE_OS_UNLINK(bcc_name);
      remove_stack_file();
    }
      
    // Open the file for writing.  First we will try to open 
    // it with the O_EXCL flag set.  This will fail if the file
    // already exists.  If this is the case, we will call open()
    // again without it and read in the header block.
    if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(bcc_fd_ = TPIE_OS_OPEN_OEXCL(bcc_name,mapping))) {
			
      // Try again, hoping the file already exists.
      if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(bcc_fd_ = TPIE_OS_OPEN_ORDWR(bcc_name,mapping))) {
        status_ = BTE_COLLECTION_STATUS_INVALID;        
       TP_LOG_FATAL_ID("open() failed to open file:");
	TP_LOG_FATAL_ID(bcc_name);
        return;
      }
      
      if (read_header(bcc_name) != BTE_ERROR_NO_ERROR) {
	status_ = BTE_COLLECTION_STATUS_INVALID;
	return;
      }

      // Check whether we need a stack.
      if (header_.used_blocks < header_.last_block - 1) {
	create_stack();
	if (freeblock_stack_->status() == BTE_STREAM_STATUS_INVALID) {
	  status_ = BTE_COLLECTION_STATUS_INVALID;
	  return;
	}
      } else 
	freeblock_stack_ = NULL;

    } else {   // The file was just created.
      
      tp_assert(header_.magic_number == BTE_COLLECTION_HEADER_MAGIC_NUMBER, "Header magic number mismatch.");
      tp_assert(header_.os_block_size == os_block_size_, "Header os_block_size mismatch.");

      header_.block_size = logical_block_factor * header_.os_block_size;

      if (write_header(bcc_name) != BTE_ERROR_NO_ERROR) {
	status_ = BTE_COLLECTION_STATUS_INVALID;
	return;
      }

      // No stack (yet). Will be created by delete if needed.
      freeblock_stack_ = NULL;

      gstats_.record(COLLECTION_CREATE);
      stats_.record(COLLECTION_CREATE);
    }
  }

#if defined(__sun__) 
  if (direct_io)
    directio(bcc_fd_, DIRECTIO_ON);
  else
    directio(bcc_fd_, DIRECTIO_OFF);
#endif

  gstats_.record(COLLECTION_OPEN);
  stats_.record(COLLECTION_OPEN);
}

#if defined(__sun__) 
template<class BIDT>
bool BTE_collection_base<BIDT>::direct_io = false;
#endif


template<class BIDT>
BTE_err BTE_collection_base<BIDT>::read_header(char* bcc_name) {

  char * tmp_buffer = new char[os_block_size_];

  if (TPIE_OS_LSEEK(bcc_fd_, 0, TPIE_OS_FLAG_SEEK_SET) != 0) {
   TP_LOG_FATAL_ID("Failed to lseek in file:");
   TP_LOG_FATAL_ID(bcc_name);
    return BTE_ERROR_IO_ERROR;
  }

  if (TPIE_OS_READ(bcc_fd_, (char *)tmp_buffer, os_block_size_) != (int)os_block_size_) {
   TP_LOG_FATAL_ID("Failed to read() in file:");
   TP_LOG_FATAL_ID(bcc_name);
    return BTE_ERROR_IO_ERROR;
  }

  file_pointer = os_block_size_;

  memcpy((void *) &header_, (const void *) tmp_buffer, 
	 sizeof(BTE_collection_header));
  delete [] tmp_buffer;

  // Do some error checking on the header, such as to make sure that
  // it has the correct header version, block size etc.
  if (header_.magic_number != BTE_COLLECTION_HEADER_MAGIC_NUMBER || 
      header_.os_block_size != os_block_size_) {
   TP_LOG_FATAL_ID("Invalid header in file: ");
   TP_LOG_FATAL_ID(bcc_name);
    return BTE_ERROR_BAD_HEADER;
  }
    
  TPIE_OS_OFFSET lseek_retval;
  // Some more error checking.
  if ((lseek_retval = TPIE_OS_LSEEK(bcc_fd_, 0, TPIE_OS_FLAG_SEEK_END)) != bid_to_file_offset(header_.total_blocks)) {
   TP_LOG_FATAL_ID("File length mismatch for:");
   TP_LOG_FATAL_ID(bcc_name);
   TP_LOG_FATAL("\tReturn value of seek (to end): ");
   TP_LOG_FATAL(lseek_retval);
   TP_LOG_FATAL("\n\tReturn value of bid_to_file_offset(header_.total_blocks): ");
   TP_LOG_FATAL(bid_to_file_offset(header_.total_blocks));
   TP_LOG_FATAL("\n\theader_.total_blocks: ");
   TP_LOG_FATAL(header_.total_blocks);
   TP_LOG_FATAL("\n");
    return BTE_ERROR_BAD_HEADER;
  }

  file_pointer = lseek_retval;

  return BTE_ERROR_NO_ERROR;
}


template<class BIDT>
BTE_err BTE_collection_base<BIDT>::write_header(char *bcc_name) {

  char * tmp_buffer = new char[os_block_size_];
  memcpy((void *) tmp_buffer, (const void *) &header_, 
	 sizeof(BTE_collection_header));

  if (TPIE_OS_LSEEK(bcc_fd_, 0, TPIE_OS_FLAG_SEEK_SET) != 0) {
   TP_LOG_FATAL_ID("Failed to lseek() in file:");
   TP_LOG_FATAL_ID(bcc_name);
    return BTE_ERROR_IO_ERROR;
  }

  if (TPIE_OS_WRITE(bcc_fd_, tmp_buffer, os_block_size_) != (int)os_block_size_) {
   TP_LOG_FATAL_ID("Failed to write() in file:");
   TP_LOG_FATAL_ID(bcc_name);
    return BTE_ERROR_IO_ERROR;
  }

  file_pointer = os_block_size_;

  // TP_LOG_APP_DEBUG_ID("header_.total_blocks: ");
  // TP_LOG_APP_DEBUG_ID(header_.total_blocks);

  delete [] tmp_buffer;
  return BTE_ERROR_NO_ERROR;
}


template<class BIDT>
BTE_collection_base<BIDT>::~BTE_collection_base() {

  char bcc_name[BTE_COLLECTION_PATH_NAME_LEN];
  strncpy((char *) bcc_name, base_file_name_, 
	  BTE_COLLECTION_PATH_NAME_LEN - 4);
  strcat((char *) bcc_name, BTE_COLLECTION_BLK_SUFFIX);

  // No block should be in memory at the time of destruction.
  if (in_memory_blocks_) {
   TP_LOG_WARNING_ID("In memory blocks when closing collection in:");
   TP_LOG_WARNING_ID(base_file_name_);
  }

#if defined(__sun__) 
  if (direct_io)
    directio(bcc_fd_, DIRECTIO_OFF);
#endif

  // Write the header.
  if (!read_only_)
    write_header(bcc_name);

  // Delete the stack.
  if (freeblock_stack_ != NULL) {
    freeblock_stack_->persist(per_);
    delete freeblock_stack_;
  }

  // Close the blocks file.
  if (TPIE_OS_CLOSE(bcc_fd_)) {            
   TP_LOG_FATAL_ID("Failed to close() ");
   TP_LOG_FATAL_ID(bcc_name);
    return;
  }
  
  // If necessary, remove the blocks file.
  if (per_ == PERSIST_DELETE) {
    if (read_only_) {
     TP_LOG_WARNING_ID("Read-only collection is PERSIST_DELETE");
     TP_LOG_WARNING_ID(bcc_name);
      return;
    } 

    if (TPIE_OS_UNLINK(bcc_name)) {
     TP_LOG_FATAL_ID("Failed to unlink() ");
     TP_LOG_FATAL_ID(bcc_name);
      return;
    } else {
      gstats_.record(COLLECTION_DELETE);
      stats_.record(COLLECTION_DELETE);
    }
  }

  gstats_.record(COLLECTION_CLOSE);
  stats_.record(COLLECTION_CLOSE);
}




#endif //_BTE_COLL_BASE_H
