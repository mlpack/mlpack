//
// File: bte_stream_ufs.h (formerly bte_ufs.h)
// Author: Rakesh Barve <rbarve@cs.duke.edu>
//
// $Id: bte_stream_ufs.h,v 1.17 2005/07/07 20:36:12 adanner Exp $
//
// BTE streams with blocks I/Oed using read()/write().  This particular
// implementation explicitly manages blocks, and only ever maps in one
// block at a time.  This relies on the filesystem to do lookahead. It
// is assumed for the purpose of memory calculations that for each
// block used by TPIE, the filesystem uses up another block of the
// same size.
//
// Completely different from the old bte_ufs.h since this does
// blocking like bte_mmb, only it uses read()/write() to do so.

// TODO: Get rid of or fix the LIBAIO stuff. As it is now it has no
// chance of working, since it uses the static
// BTE_STREAM_UFS_BLOCK_FACTOR, which is no longer the true
// factor. The true block factor is determined dynamically, from the
// header.
//

#ifndef _BTE_STREAM_UFS_H
#define _BTE_STREAM_UFS_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// For header's type field (85 == 'U').
#define BTE_STREAM_UFS 85

//the code for double buffering is not here..
#define UFS_DOUBLE_BUFFER 0

// Either double buffer explicitly using aio or aio can be used Darren
// style or more directly. Using it directly will probably be better,
// but right now that is not supported. (Solaris and Digital/FreeBSD
// use different aio interfaces.
#if BTE_STREAM_UFS_READ_AHEAD	
#  if !USE_LIBAIO && !UFS_DOUBLE_BUFFER
#    error BTE_STREAM_UFS_READ_AHEAD requested, but no double buff mechanism in config.
#  endif
#  define BTE_STREAM_UFS_MM_BUFFERS 2
#else
#  define BTE_STREAM_UFS_MM_BUFFERS 1
#endif

#if UFS_DOUBLE_BUFFER
#  error At present explicit DOUBLE BUFFER not supported.
#endif

// The double buffering mechanism will use lib_aio on Solaris and the
// asynch.h interface on Digital Unix and FreeBSD.  Gut feeling is
// that if file access is maintained sequential performance with both
// UFS_DOUBLE_BUFFER and USE_LIBAIO set off is best.

#if USE_LIBAIO
#  if !HAVE_LIBAIO
#    error USE_LIBAIO requested, but aio library not in configuration.
#  endif
#  if UFS_DOUBLE_BUFFER
#    error Darren-style USE_LIBAIO requested, but so is DOUBLE BUFFER
#  endif
#endif

// Get the BTE_stream_base class and related definitions.
#include <bte_stream_base.h>

// This code makes assertions and logs errors.
#include <tpie_assert.h>
#include <tpie_log.h>

// Define a sensible logical block factor, if not already defined.
#ifndef  BTE_STREAM_UFS_BLOCK_FACTOR
#  define BTE_STREAM_UFS_BLOCK_FACTOR 8
#endif


// This is a class template for the implementation of a 
// BTE stream of objects of type T such that the entire stream 
// resides on a single disk.  This version maps in only one
// block of the file at a time. The striped_stream class, such
// that it is comprised of several single disk streams, has  
// a member function that is a friend of this class.
template < class T > class BTE_stream_ufs: public BTE_stream_base < T > {
private:

    // Descriptor of the mapped file.
    TPIE_OS_FILE_DESCRIPTOR fd;

    TPIE_OS_SIZE_T os_block_size_;

    int itemsize_div_blocksize;

    // Offset of the current item in the file. This is the logical
    // offset of the item within the file, that is, the place we would
    // have to lseek() to in order to read() or write() the item if we
    // were using ordinary (i.e. non-mmap()) file access methods.
    TPIE_OS_OFFSET f_offset;

    // [tavi 01/27/02]
    // This is the position in the file where the pointer is. We can
    // save some lseek() calls by maintaining this.
    TPIE_OS_OFFSET file_pointer;

    // Offset just past the end of the last item in the stream. If this
    // is a substream, we can't write here or anywhere beyond.
    TPIE_OS_OFFSET f_eos;

    // Beginning of the file.  Can't write before here.
    TPIE_OS_OFFSET f_bos;

    TPIE_OS_OFFSET f_filelen;

    // A pointer to the mapped in header block for the stream.
    BTE_stream_header *header;

    // The current item (mapped in)
    T *current;
    // A pointer to the beginning of the currently mapped block.
    T *curr_block;

    // Non-zero if current points to a valid, mapped in block.
    int block_valid;

    // If block_valid is one, then block_dirty is 1 if and only if
    // mapped block is dirty; obviously block_dirty is always 0 for
    // r_only streams.
    int block_dirty;

    // When block_valid is one, this is the Offset of curr_block in the
    // underlying Unix file.
    TPIE_OS_OFFSET curr_block_file_offset;	

    TPIE_OS_SIZE_T blocksize_items;

    // A place to cache OS error values. It is normally set after each
    // call to the OS.
    int os_errno;
    // The file name.
    char path[BTE_STREAM_PATH_NAME_LEN];

#if UFS_DOUBLE_BUFFER
    // for use in double buffering, when one is implemented using
    // the aio interface.
    T *next_block;		// ptr to next block 
    TPIE_OS_OFFSET f_next_block;		// position of next block
    int have_next_block;		// is next block mapped?

#endif	/* UFS_DOUBLE_BUFFER */

#if USE_LIBAIO
    // A buffer to read the first word of each OS block in the next logical
    // block for read ahead.
    int read_ahead_buffer[BTE_STREAM_UFS_BLOCK_FACTOR];
    // Results of asyncronous I/O.
    aio_result_t aio_results[BTE_STREAM_UFS_BLOCK_FACTOR];
#endif	/* USE_LIBAIO */

#if BTE_STREAM_UFS_READ_AHEAD
    // Read ahead into the next logical block.
    void read_ahead (void);
#endif

    BTE_stream_header *map_header (void);

    inline BTE_err validate_current (void);
    inline BTE_err invalidate_current (void);

    BTE_err map_current (void);
    BTE_err unmap_current (void);

    inline BTE_err advance_current (void);

    inline TPIE_OS_OFFSET item_off_to_file_off (TPIE_OS_OFFSET item_off) const;
    inline TPIE_OS_OFFSET file_off_to_item_off (TPIE_OS_OFFSET item_off) const;

 protected:
  using BTE_stream_base<T>::remaining_streams;
  using BTE_stream_base<T>::gstats_;
  using BTE_stream_base<T>::status_;
  using BTE_stream_base<T>::stats_;
  using BTE_stream_base<T>::substream_level;
  using BTE_stream_base<T>::per;
  using BTE_stream_base<T>::r_only;
  
 public:
  using BTE_stream_base<T>::os_block_size;
  using BTE_stream_base<T>::check_header;
  using BTE_stream_base<T>::init_header;

 public:
    // Constructor.
    // [tavi 01/09/02] Careful with the lbf (logical block factor)
    // parameter. I introduced it in order to avoid errors when reading
    // a stream having a different block factor from the default, but
    // this make cause errors in applications. For example, the
    // AMI_partition_and merge computes memory requirements of temporary
    // streams based on the memory usage of the INPUT stream, However,
    // the input stream may have different block size from the temporary
    // streams created later. Until these issues are addressed, the
    // usage of lbf is discouraged.
    BTE_stream_ufs (const char *dev_path, BTE_stream_type st,
		    TPIE_OS_SIZE_T lbf = BTE_STREAM_UFS_BLOCK_FACTOR);

    // A substream constructor.
    BTE_stream_ufs (BTE_stream_ufs * super_stream,
		    BTE_stream_type st, TPIE_OS_OFFSET sub_begin, TPIE_OS_OFFSET sub_end);

    // A psuedo-constructor for substreams.
    BTE_err new_substream (BTE_stream_type st, TPIE_OS_OFFSET sub_begin,
			   TPIE_OS_OFFSET sub_end,
			   BTE_stream_base < T > **sub_stream);

    // Query memory usage
    BTE_err main_memory_usage (size_t * usage, MM_stream_usage usage_type);

    // Return the number of items in the stream.
    TPIE_OS_OFFSET stream_len () const;

    // Return the path name in newly allocated space.
    BTE_err name (char **stream_name);

    // Move to a specific position in the stream.
    BTE_err seek (TPIE_OS_OFFSET offset);

    // Return the current position in the stream.
    TPIE_OS_OFFSET tell () const;

    // Truncate the stream.
    BTE_err truncate (TPIE_OS_OFFSET offset);

    // Destructor
    ~BTE_stream_ufs ();

    B_INLINE BTE_err read_item (T ** elt);
    B_INLINE BTE_err write_item (const T & elt);

    TPIE_OS_OFFSET chunk_size (void) const;
};

// This constructor creates a stream whose contents are taken from the
// file whose path is given.
template < class T >
BTE_stream_ufs < T >::BTE_stream_ufs (const char *dev_path,
				      BTE_stream_type st,
				      TPIE_OS_SIZE_T lbf) {

    status_ = BTE_STREAM_STATUS_NO_STATUS;

    // Check if we have available streams. Don't decrease the number
    // yet, since we may encounter an error.
    if (remaining_streams <= 0) {
	status_ = BTE_STREAM_STATUS_INVALID;
	TP_LOG_FATAL_ID ("BTE internal error: cannot open more streams.");
	return;
    }

    // Cache the path name
    if (strlen (dev_path) > BTE_STREAM_PATH_NAME_LEN - 1) {
	status_ = BTE_STREAM_STATUS_INVALID;
	TP_LOG_FATAL_ID ("Path name \"" << dev_path << "\" too long.");
	return;
    }

    strncpy (path, dev_path, BTE_STREAM_PATH_NAME_LEN);

    // Cache the OS block size.
    os_block_size_ = os_block_size();

    // This is a top level stream
    substream_level = 0;

    per = PERSIST_PERSISTENT;
    block_valid = 0;
    block_dirty = 0;
    // A field to remember the file offset of mapped in block.
    curr_block_file_offset = 0;
    curr_block = current = NULL;
    f_offset = f_bos = os_block_size_;
    // To be on the safe side, set this to -1. It will be set to the
    // right value by map_header(), below.
    file_pointer = -1; 

    // Decrease the number of available streams.
    remaining_streams--;

    switch (st) {
    case BTE_READ_STREAM:

	r_only = 1;

	// Open the file for reading.
	if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(fd = TPIE_OS_OPEN_ORDONLY(path))) {
	    status_ = BTE_STREAM_STATUS_INVALID;
	    os_errno = errno;
	   TP_LOG_FATAL_ID ("open() failed to open " << path);
	   TP_LOG_FATAL_ID (strerror (os_errno));
	    // [tavi 01/07/02] Commented this out. Just because the file is
	    // unreadable is no reason to crash.
	    //assert (0); 
	    return;
	}

	header = map_header ();
	if (check_header (header) < 0) {
	    status_ = BTE_STREAM_STATUS_INVALID;
	    return;
	}
	// Some more checking, specific to this stream type.
	if (header->type != BTE_STREAM_UFS) {
	   TP_LOG_WARNING_ID("Using UFS stream implem. on another type of stream.");
	   TP_LOG_WARNING_ID("Stream implementations may not be compatible.");
	}
	if ((header->block_size % os_block_size_ != 0) || 
	    (header->block_size == 0)) {
	    status_ = BTE_STREAM_STATUS_INVALID;
	   TP_LOG_FATAL_ID ("header: incorrect logical block size;");
	   TP_LOG_FATAL_ID ("expected multiple of OS block size.");
	    return;
	}
	if (header->block_size != BTE_STREAM_UFS_BLOCK_FACTOR * os_block_size_) {
	   TP_LOG_WARNING_ID("Stream has different block factor than the default.");
	   TP_LOG_WARNING_ID("This may cause problems in some existing applications.");
	}

	blocksize_items = header->block_size / sizeof (T);
	itemsize_div_blocksize = (header->block_size % sizeof (T) == 0);

	// Set the eos marker appropriately.
	f_eos = item_off_to_file_off (header->item_logical_eof);

	if (header->item_logical_eof >= 1) {
	    if (f_eos - item_off_to_file_off (header->item_logical_eof - 1) -
		sizeof (T) > 0) {
		// Meaning, 1. sizeof (T) does not divide the logical
		// blocksize. 2. the last item in the stream is the last
		// item that could have been placed on its logical block
		// (so that the valid file offset as far as TPIE goes, is
		// the beginning of a new block and so strictly greater
		// than the byte offset at which the last item ends). In
		// this situation, after reading the last item and f_offset
		// gets incremented, it is strictly less than f_eos; as a
		// result the check (f_eos <= f_offset)? in ::read_item()
		// gets beaten when it shouldn't.  To remedy, we simply
		// reset f_eos in this circumstance to be just past the
		// last item's byte offset.

		f_eos = item_off_to_file_off (header->item_logical_eof - 1) +
		    sizeof (T);
	    }
	}
	break;

    case BTE_WRITE_STREAM:
    case BTE_WRITEONLY_STREAM:
    case BTE_APPEND_STREAM:

	r_only = 0;

	// Open the file for writing.  First we will try to open 
	// is with the O_EXCL flag set.  This will fail if the file
	// already exists.  If this is the case, we will call open()
	// again without it and read in the header block.
	if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(fd = TPIE_OS_OPEN_OEXCL(path))) {

	    // Try again, hoping the file already exists.
	    if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(fd = TPIE_OS_OPEN_ORDWR(path))) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;
		TP_LOG_FATAL_ID ("open() failed to open " << path);
		TP_LOG_FATAL_ID (strerror (os_errno));
		return;
	    }
	    // The file already exists, so read the header.
	    header = map_header ();
	    if (check_header (header) < 0) {
		status_ = BTE_STREAM_STATUS_INVALID;
		return;
	    }
	    // Some more checking, specific to this stream.
	    if (header->type != BTE_STREAM_UFS) {
		TP_LOG_WARNING_ID("Using UFS stream implem. on another type of stream.");
		TP_LOG_WARNING_ID("Stream implementations may not be compatible.");
	    }
	    if ((header->block_size % os_block_size_ != 0) || 
		(header->block_size == 0)) {
		status_ = BTE_STREAM_STATUS_INVALID;
 		TP_LOG_FATAL_ID("Header: incorrect logical block size;");
		TP_LOG_FATAL_ID("Expected multiple of OS block size.");
		return;
	    }
	    if (header->block_size != BTE_STREAM_UFS_BLOCK_FACTOR * os_block_size_) {
		TP_LOG_WARNING_ID("Stream has different block factor than the default;");
		TP_LOG_WARNING_ID("\tStream block factor: " << (TPIE_OS_LONGLONG)header->block_size/os_block_size_);
		TP_LOG_WARNING_ID("\tDefault block factor: " << BTE_STREAM_UFS_BLOCK_FACTOR);
		TP_LOG_WARNING_ID("This may cause problems in some existing applications.");
	    }

	    blocksize_items = header->block_size / sizeof (T);
	    itemsize_div_blocksize = (header->block_size % sizeof (T) == 0);

	    f_eos = item_off_to_file_off (header->item_logical_eof);

	    if (header->item_logical_eof >= 1) {
		if (f_eos - item_off_to_file_off (header->item_logical_eof - 1) -
		    sizeof (T) > 0) {
		    // Meaning, 1. sizeof (T) does not divide the logical
		    // blocksize. 2. the last item in the stream is the last
		    // item that could have been placed on its logical block
		    // (so that the valid file offset as far as TPIE goes,
		    // is the beginning of a new block and so strictly
		    // greater than the byte offset at which the last item
		    // ends). In this situation, after reading the last
		    // item and f_offset gets incremented, it is strictly
		    // less than f_eos; as a result the check (f_eos <=
		    // f_offset)? in ::read_item() gets beaten when it
		    // shouldn't. To remedy, we simply reset f_eos in this
		    // circumstance to be just past the last item's byte
		    // offset.
		    f_eos =  item_off_to_file_off (header->item_logical_eof - 1) +
			sizeof (T);
		}
	    }

	    if (st == BTE_APPEND_STREAM) {
		f_offset = f_eos;
	    }

	} else {	// The file was just created.

	    // Create and map in the header. File does not exist, so
	    // first establish a mapping and then write into the file via
	    // the mapping.
	    header = map_header ();
	    if (header == NULL) {
		status_ = BTE_STREAM_STATUS_INVALID;
		return;
	    }
	    init_header (header);

	    if (lbf == 0) {
		lbf = 1;
		TP_LOG_WARNING_ID("Block factor 0 requested. Using 1 instead.");
	    }
	    // Set the logical block size.
	    header->block_size = lbf * os_block_size_;
	    // Set the type.
	    header->type = BTE_STREAM_UFS;

	    blocksize_items = header->block_size / sizeof (T);
	    itemsize_div_blocksize = (header->block_size % sizeof (T) == 0);
	 
	    f_eos = os_block_size_;
	    gstats_.record(STREAM_CREATE);
	    stats_.record(STREAM_CREATE);
	}

	break;
    }				// end of switch

    // We can't handle streams of large objects.
    if (sizeof (T) > header->block_size) {
	status_ = BTE_STREAM_STATUS_INVALID;
	TP_LOG_FATAL_ID ("Object is too big (object size/block size):");
	TP_LOG_FATAL_ID (sizeof(T));
	TP_LOG_FATAL_ID ((TPIE_OS_LONGLONG)header->block_size);
	return;
    }

#if UFS_DOUBLE_BUFFER
    next_block = NULL;
    f_next_block = 0;
    have_next_block = 0;
#endif


    // Memory-usage for the object, base class, header and the stream buffers
    // are registered automatically by Darren's modified new() function.
    f_filelen = TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_END);
    file_pointer = f_filelen;
    gstats_.record(STREAM_OPEN);
    stats_.record(STREAM_OPEN);
}


// A substream constructor.
// sub_begin is the item offset of the first item in the stream.
// sub_end is the item offset that of the last item in the stream.
// Thus, f_eos in the new substream will be set to point one item beyond
// this.
//
// For example, if a stream contains [A,B,C,D,...] then substream(1,3)
// will contain [B,C,D].
template < class T >
BTE_stream_ufs < T >::BTE_stream_ufs (BTE_stream_ufs * super_stream,
				      BTE_stream_type st,
				      TPIE_OS_OFFSET sub_begin, TPIE_OS_OFFSET sub_end) {

    status_ = BTE_STREAM_STATUS_NO_STATUS;

    // Reduce the number of streams avaialble.
    if (remaining_streams <= 0) {
	status_ = BTE_STREAM_STATUS_INVALID;
	TP_LOG_FATAL_ID ("BTE error: cannot open more streams.");
	return;
    }

    if (super_stream->status_ == BTE_STREAM_STATUS_INVALID) {
	status_ = BTE_STREAM_STATUS_INVALID;
	TP_LOG_FATAL_ID ("BTE error: super stream is invalid.");
	return;
    }

    if (super_stream->r_only && (st != BTE_READ_STREAM)) {
	status_ = BTE_STREAM_STATUS_INVALID;
	TP_LOG_FATAL_ID
	    ("BTE error: super stream is read only and substream is not.");
	return;
    }

    // If you are going to access a substream of a previously created
    // (super)stream we want to make sure that the superstream 's
    // currently valid block, if any, is committed to the underlying
    // Unix file. Note that with memory mapped implementation such a
    // "committing" is automatic but in our case we need to keep track
    // of such things.
    if (!super_stream->r_only && super_stream->block_valid) {

	super_stream->unmap_current ();

	if (super_stream->status_ == BTE_STREAM_STATUS_INVALID) {
	    status_ = BTE_STREAM_STATUS_INVALID;
	   TP_LOG_FATAL_ID ("BTE internal error: super stream is invalid.");
	    return;
	}
    }

    remaining_streams--;

    // Copy the relevant fields from the super_stream.
    
    r_only = super_stream->r_only;
    strncpy (path, super_stream->path, BTE_STREAM_PATH_NAME_LEN);
	os_block_size_ = super_stream->os_block_size_;
    blocksize_items = super_stream->blocksize_items;
    itemsize_div_blocksize = super_stream->itemsize_div_blocksize;
    header = super_stream->header;
    substream_level = super_stream->substream_level + 1;

	//Each substream should have a local file descriptor 
	//so file_pointer and fd position match
	//Only READ and WRITE streams allowed
	switch(st){
	  case BTE_READ_STREAM:
		fd=TPIE_OS_OPEN_ORDONLY(path);
		break;
	  case BTE_WRITE_STREAM:
		//file better exist if super_stream exists
		fd=TPIE_OS_OPEN_ORDWR(path);
		break;
	  default:
	    status_ = BTE_STREAM_STATUS_INVALID;
	   TP_LOG_FATAL_ID ("BTE internal error: Invalid subtream type.");
	    return;
    }

	if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(fd)) {
	  status_ = BTE_STREAM_STATUS_INVALID;
	  os_errno = errno;
	 TP_LOG_FATAL_ID ("open() failed to open " << path);
	 TP_LOG_FATAL_ID (strerror (os_errno));
	  assert(0);
	  return;
	}

    per = PERSIST_PERSISTENT;

    // The arguments sub_start and sub_end are logical item positions
    // within the stream.  We need to convert them to offsets within
    // the stream where items are found.

    TPIE_OS_OFFSET super_item_begin = file_off_to_item_off (super_stream->f_bos);

    f_bos = item_off_to_file_off (super_item_begin + sub_begin);
    f_eos = item_off_to_file_off (super_item_begin + sub_end + 1);

    tp_assert (f_bos <= f_eos, "bos beyond eos");	// sanity check

    if (super_item_begin + sub_end + 1 >= 1) {
	if (f_eos - item_off_to_file_off (super_item_begin + sub_end) -
	    sizeof (T) > 0) {
	    // Meaning, 1. sizeof (T) does not divide the logical
	    // blocksize. 2. the last item in the stream is the last item
	    // that could have been placed on its logical block (so that
	    // the valid file offset as far as TPIE goes, is the beginning
	    // of a new block and so strictly greater than the byte offset
	    // at which the last item ends.)  In this situation, after
	    // reading the last item and f_offset gets incremented, it is
	    // strictly less than f_eos; as a result the check (f_eos <=
	    // f_offset)? in ::read_item() gets beaten when it shouldn't.
	    // To remedy, we simply reset f_eos in this circumstance to be
	    // just past the last item's byte offset.
	    f_eos = item_off_to_file_off (super_item_begin + sub_end) +
		sizeof (T);
	}
    }

    tp_assert (f_bos <= f_eos, "bos beyond eos");	// sanity check

    f_filelen = super_stream->f_filelen;

    if (f_eos > super_stream->f_eos) {
	status_ = BTE_STREAM_STATUS_INVALID;
	TP_LOG_FATAL_ID ("BTE internal error: reached beyond super stream eof.");
	return;
    }

    f_offset = f_bos;
    file_pointer = -1; // I don't know where the file pointer is.
    current = curr_block = NULL;
    block_valid = 0;
    block_dirty = 0;
    curr_block_file_offset = 0;

#if UFS_DOUBLE_BUFFER
    next_block = NULL;
    f_next_block = 0;
    have_next_block = 0;
#endif

    // Register memory_usage for the object corresp to the substream.
    gstats_.record(STREAM_OPEN);
    gstats_.record(SUBSTREAM_CREATE);
    stats_.record(STREAM_OPEN);
    stats_.record(SUBSTREAM_CREATE);
}

// A psuedo-constructor for substreams.  This serves as a wrapper for
// the constructor above in order to get around the fact that one
// cannot have virtual constructors.
template < class T >
BTE_err BTE_stream_ufs < T >::new_substream (BTE_stream_type st,
					     TPIE_OS_OFFSET sub_begin,
					     TPIE_OS_OFFSET sub_end,
					     BTE_stream_base < T >
					     **sub_stream) {
    // Check permissions.
    if ((st != BTE_READ_STREAM) && ((st != BTE_WRITE_STREAM) || r_only)) {
	*sub_stream = NULL;
	return BTE_ERROR_PERMISSION_DENIED;
    }

    tp_assert (((st == BTE_WRITE_STREAM) && !r_only) ||
	       (st == BTE_READ_STREAM),
	       "Bad things got through the permisssion checks.");

    BTE_stream_ufs < T > *sub =
	new BTE_stream_ufs < T > (this, st, sub_begin, sub_end);

    *sub_stream = (BTE_stream_base < T > *) sub;

    return BTE_ERROR_NO_ERROR;
}

template < class T > BTE_stream_ufs < T >::~BTE_stream_ufs (void) {

    // If the stream is already invalid for some reason, then don't
    // worry about anything.
    if (status_ == BTE_STREAM_STATUS_INVALID) {
	TP_LOG_WARNING_ID ("BTE internal error: invalid stream in destructor.");
	return;
    }

    // Increase the number of streams avaialble.
    if (remaining_streams >= 0) {
	remaining_streams++;
    }
    gstats_.record(STREAM_DELETE);
    stats_.record(STREAM_DELETE);

    // If this is writable and not a substream, then put the logical
    // eos back into the header before unmapping it.
    if (!r_only && !substream_level) {
	header->item_logical_eof = file_off_to_item_off (f_eos);
    }

    // Unmap the current block if necessary.
    if (block_valid) {
	unmap_current ();
    }

    // If this is not a substream then cleanup.
    if (!substream_level) {
      // If a writeable stream, write back the header.  But only if
      // the stream is persistent. Otherwise, don't waste time with
      // the system calls.
	if (!r_only && per != PERSIST_DELETE) {
	    if (TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_SET) != 0) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;
		TP_LOG_FATAL_ID ("lseek() failed to move past header of " << path);
		TP_LOG_FATAL_ID (strerror (os_errno));
		// [tavi 01/07/02] Commented this out. Why panic?
		//assert (0);
		// TODO: Should we really return? If we do, we have memory leaks.
		return;
	    }
	    if (TPIE_OS_WRITE (fd, (char *) header, sizeof (BTE_stream_header))
		!= sizeof (BTE_stream_header)) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;
		TP_LOG_FATAL_ID ("write() failed during stream destruction for "
			      << path);
		TP_LOG_FATAL_ID (strerror (os_errno));
		// [tavi 01/07/02] Commented this out. Why panic?
		//assert (0);
		// TODO: Should we really return? If we do, we have memory leaks.
		return;
	    }
	    // Invalidate the cached file pointer.
	    file_pointer = -1;
	}

	if (header)
	    delete header;

	if (TPIE_OS_CLOSE (fd)) {
	    os_errno = errno;
	   TP_LOG_FATAL_ID ("Failed to close() " << path);
	   TP_LOG_FATAL_ID (strerror (os_errno));
	    // [tavi 01/07/02] Commented this out. Why panic?
	    //assert (0);
	    return;
	}

	// If it should not persist, unlink the file.
	if (per == PERSIST_DELETE) {
	    if (r_only) {
		TP_LOG_WARNING_ID("PERSIST_DELETE for read-only stream in " << path);
            } else  {
	      if (TPIE_OS_UNLINK (path)) {
		os_errno = errno;
		TP_LOG_WARNING_ID ("unlink failed during destruction of:");
		TP_LOG_WARNING_ID (path);
		TP_LOG_WARNING_ID (strerror (os_errno));
	      } else {
		gstats_.record(STREAM_DELETE);
		stats_.record(STREAM_DELETE);
	      }
	    }
	}
  } else {				// end of if (!substream_level) 
		//Each substream has its own file descriptor so close it. 
	  if (TPIE_OS_CLOSE (fd)) {
	    os_errno = errno;
	   TP_LOG_FATAL_ID ("Failed to close() substream" << path);
	   TP_LOG_FATAL_ID (strerror (os_errno));
	    return;
		}
	  gstats_.record(SUBSTREAM_DELETE);
	  stats_.record(SUBSTREAM_DELETE);
  }

    if (curr_block) {
	delete [] curr_block; // should be vector delete -RW

	// If you really want to be anal about memory calculation
	// consistency then if BTE_IMPLICIT_FS_READAHEAD flag is set you
	// should register a memory deallocation of header->block_size AT
	// THIS POINT of time in code.  At present, since we havent
	// registered allocation for these ``implicitly read-ahead''
	// blocks, we don't register the dealloc either.
    }
#if UFS_DOUBLE_BUFFER
    //Have to think this out since if UFS_DOUBLE_BUFFERING is implemented
    //there is the possibility that the aio_read for the next block is
    //ongoing at the time of the destruction, in which case trying to
    //delete next_block may cause a run-time error. Most probably
    // the aio read op may have to be suspended if ongoing. 
    if (next_block)
	delete [] next_block;	// use vector delete -RW
#endif

    gstats_.record(STREAM_CLOSE);
    stats_.record(STREAM_CLOSE);
}

template < class T >
B_INLINE BTE_err BTE_stream_ufs < T >::read_item (T ** elt) {

    BTE_err bte_err;
   
    // Make sure we are not currently at the EOS.
    if (f_offset >= f_eos) {
	tp_assert (f_eos == f_offset, "Can't read past eos.");
	return BTE_ERROR_END_OF_STREAM;
    }
    // Validate the current block.
    if ((bte_err = validate_current ()) != BTE_ERROR_NO_ERROR) {
	return bte_err;
    }

    // Check and make sure that the current pointer points into the
    // current block.
    tp_assert (((unsigned int) ((char *) current - (char *) curr_block) <=
		(unsigned int) (header->block_size - sizeof (T))),
	       "current is past the end of the current block");
    tp_assert (((char *) current - (char *) curr_block >= 0),
	       "current is before the begining of the current block");

    gstats_.record(ITEM_READ);
    stats_.record(ITEM_READ);

    // Read
    *elt = current;

    // Advance the current pointer.
    advance_current ();

    // If we are in a substream, there should be no way for f_current
    // to pass f_eos.
    tp_assert (!substream_level || (f_offset <= f_eos),
	       "Got past eos in a substream.");

    return BTE_ERROR_NO_ERROR;
}

template < class T >
B_INLINE BTE_err BTE_stream_ufs < T >::write_item (const T & elt) {

    BTE_err bte_err;

    // This better be a writable stream.
    if (r_only) {
	return BTE_ERROR_READ_ONLY;
    }
    // Make sure we are not currently at the EOS of a substream.
    if (substream_level && (f_eos <= f_offset)) {
	tp_assert (f_eos == f_offset, "Went too far in a substream.");
	return BTE_ERROR_END_OF_STREAM;
    }
    // Validate the current block.
    if ((bte_err = validate_current ()) != BTE_ERROR_NO_ERROR) {
	return bte_err;
    }

    // Check and make sure that the current pointer points into the current
    // block.
    tp_assert (((unsigned int) ((char *) current - (char *) curr_block) <=
		(unsigned int) (header->block_size - sizeof (T))),
	       "current is past the end of the current block");
    tp_assert (((char *) current - (char *) curr_block >= 0),
	       "current is before the begining of the current block");

    gstats_.record(ITEM_WRITE);
    stats_.record(ITEM_WRITE);

    // Write.
    *current = elt;
    block_dirty = 1;

    // Advance the current pointer.
    advance_current ();

    // If we are in a substream, there should be no way for f_current to
    // pass f_eos.
    tp_assert (!substream_level || (f_offset <= f_eos),
	       "Got past eos in a substream.");

    // If we moved past eos, then update eos unless we are in a
    // substream, in which case EOS will be returned on the next call.
    if ((f_offset > f_eos) && !substream_level) {
	// disable the assertion below because it is violated when
	// the end of a block is reached and the item size does not
	// divide the block size completely (so there is some space left)
	// tp_assert(f_offset == f_eos + sizeof(T), "Advanced too far somehow.");
	f_eos = f_offset;
    }

    return BTE_ERROR_NO_ERROR;
}

// Query memory usage
// Note that in a substream we do not charge for the memory used by
// the header, since it is accounted for in the 0 level superstream.
template < class T >
BTE_err BTE_stream_ufs < T >::main_memory_usage (size_t * usage,
						 MM_stream_usage
						 usage_type)
{
    switch (usage_type) {
      case MM_STREAM_USAGE_OVERHEAD:
        //sizeof(*this) includes base class. 
        //header is allocated dynamically, but always allocated, 
        //even for substreams. Don't forget space overhead per
        //"new" on (class, base class, header) 
        *usage = sizeof(*this) + sizeof(BTE_stream_header) +
                 3*MM_manager.space_overhead();
        break;
      case MM_STREAM_USAGE_BUFFER: 
        //space used by buffers, when allocated
        *usage = BTE_STREAM_UFS_MM_BUFFERS * header->block_size +
                 MM_manager.space_overhead();
        break;
      case MM_STREAM_USAGE_CURRENT:
        //overhead + buffers (if in use)
        *usage = sizeof(*this) +  sizeof(BTE_stream_header) +
           3*MM_manager.space_overhead() + 
           ((curr_block == NULL) ? 0 : (BTE_STREAM_UFS_MM_BUFFERS *
            header->block_size + MM_manager.space_overhead()));
        break;
      case MM_STREAM_USAGE_MAXIMUM:
      case MM_STREAM_USAGE_SUBSTREAM:
        *usage = sizeof(*this) +  sizeof(BTE_stream_header) +
           BTE_STREAM_UFS_MM_BUFFERS * header->block_size +
           4*MM_manager.space_overhead();
        break;
    }
    return BTE_ERROR_NO_ERROR;
}

// Return the number of items in the stream.
template < class T > 
TPIE_OS_OFFSET BTE_stream_ufs < T >::stream_len (void) const {
    return file_off_to_item_off (f_eos) - file_off_to_item_off (f_bos);
};

// Return the path name in newly allocated space.
template < class T >
BTE_err BTE_stream_ufs < T >::name (char **stream_name) {

    TPIE_OS_SIZE_T len = (TPIE_OS_SIZE_T)strlen (path);

    tp_assert (len < BTE_STREAM_PATH_NAME_LEN, "Path length is too long.");

    char *new_path = new char[len + 1];

    strncpy (new_path, path, len + 1);
    *stream_name = new_path;

    return BTE_ERROR_NO_ERROR;
}

// Move to a specific position.
template < class T > 
BTE_err BTE_stream_ufs < T >::seek (TPIE_OS_OFFSET offset) {

    BTE_err be;
    TPIE_OS_OFFSET new_offset;

    if ((offset < 0) ||
	(offset > file_off_to_item_off (f_eos) - 
	 file_off_to_item_off (f_bos))) {
	TP_LOG_WARNING_ID ("seek() out of range (off/bos/eos)");
	TP_LOG_WARNING_ID (offset);
	TP_LOG_WARNING_ID (file_off_to_item_off (f_bos));
	TP_LOG_WARNING_ID (file_off_to_item_off (f_eos));
	return BTE_ERROR_OFFSET_OUT_OF_RANGE;
    }

    // Compute the new offset.
    new_offset = item_off_to_file_off (file_off_to_item_off (f_bos) + offset);

    if (((size_t) ((char *) current - (char *) curr_block) >=
	 header->block_size)
	|| (((new_offset - os_block_size_) / header->block_size) !=
	    ((f_offset - os_block_size_) / header->block_size))) {
	if (block_valid && ((be = unmap_current ()) != BTE_ERROR_NO_ERROR)) {
	    return be;
	}
    } else {
	if (block_valid) {

	    // We have to adjust current.
	    register TPIE_OS_OFFSET internal_block_offset;

	    internal_block_offset = file_off_to_item_off (new_offset) % 
		blocksize_items;
	    current = curr_block + internal_block_offset;
	}
    }

    f_offset = new_offset;

    gstats_.record(ITEM_SEEK);
    stats_.record(ITEM_SEEK);
    return BTE_ERROR_NO_ERROR;
}

template < class T > 
TPIE_OS_OFFSET BTE_stream_ufs < T >::tell() const {
  return file_off_to_item_off(f_offset);
}

// Truncate the stream.
template < class T > 
BTE_err BTE_stream_ufs < T >::truncate (TPIE_OS_OFFSET offset) {

    BTE_err be;
    TPIE_OS_OFFSET new_offset;
    TPIE_OS_OFFSET block_offset;

    // Sorry, we can't truncate a substream.
    if (substream_level) {
	return BTE_ERROR_STREAM_IS_SUBSTREAM;
    }

    if (offset < 0) {
	return BTE_ERROR_OFFSET_OUT_OF_RANGE;
    }
    // Compute the new offset
    new_offset = item_off_to_file_off (file_off_to_item_off (f_bos) + offset);

    // If it is not in the same block as the current position then
    // invalidate the current block.
    // We also need to check that we have the correct block mapped in (f_offset
    // does not always point into the current block!) - see comment in seek()
    if (((unsigned int) ((char *) current - (char *) curr_block) >=
	 header->block_size)
	|| (((new_offset - os_block_size_) / header->block_size) !=
	    ((f_offset - os_block_size_) / header->block_size))) {
	if (block_valid && ((be = unmap_current ()) != BTE_ERROR_NO_ERROR)) {
	    return be;
	}
    }
    // If it is not in the same block as the current end of stream
    // then truncate the file to the end of the new last block.
    if (((new_offset - os_block_size_) / header->block_size) !=
	((f_eos - os_block_size_) / header->block_size)) {

	// Determine the offset of the block that new_offset is in.
	block_offset = ((new_offset - os_block_size_) / header->block_size)
	    * header->block_size + os_block_size_;
	f_filelen = block_offset + header->block_size;
	if (TPIE_OS_FTRUNCATE (fd, block_offset + header->block_size)) {
	    os_errno = errno;
	   TP_LOG_FATAL_ID ("Failed to ftruncate() to the new end of " << path);
	   TP_LOG_FATAL_ID (strerror (os_errno));
	    return BTE_ERROR_OS_ERROR;
	}
	// Invalidate the file pointer.
	file_pointer = -1;
    }
    // Reset the current position to the end.    
    f_offset = f_eos = new_offset;

    return BTE_ERROR_NO_ERROR;
}

// Map in the header from the file.  This assumes that the path
// has been cached in path and that the file has been opened and
// fd contains a valid descriptor.
template < class T >
BTE_stream_header * BTE_stream_ufs < T >::map_header (void) {

    TPIE_OS_OFFSET file_end;
    BTE_stream_header *ptr_to_header;

    // If the underlying file is not at least long enough to contain
    // the header block, then, assuming the stream is writable, we have
    // to create the space on disk by doing an explicit write().  
    if ((file_end = TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_END)) < (TPIE_OS_OFFSET) os_block_size_) {
	if (r_only) {
	    status_ = BTE_STREAM_STATUS_INVALID;
	   TP_LOG_FATAL_ID ("No header block in read only stream " << path);
	    return NULL;
	} else {

	    // A writable stream, but it doesn't have a header block,
	    // which means the file was just created and we have to leave
	    // space for the header block at the beginning of the fille.
	    // In this case we choose simply to allocate space for header
	    // fields and return a pointer to ufs_stream_header but first
	    // we write a dummy os_block_size_ sized block at the
	    // beginning of the file.  This will trigger off sequential
	    // write optimizations that are useful unless non-sequential
	    // accesses to data are made.

	    char *tmp_buffer = new char[os_block_size_];

	    if (file_end != 0) {
	      if (TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_SET) != 0) {
		os_errno = errno;
		TP_LOG_FATAL_ID ("Failed to lseek() in stream " << path);
		TP_LOG_FATAL_ID (strerror (os_errno));
		return NULL;
	      }
	    }

	    if (TPIE_OS_WRITE (fd, tmp_buffer, os_block_size_) !=
		(TPIE_OS_SSIZE_T) os_block_size_) {
		os_errno = errno;
		TP_LOG_FATAL_ID ("Failed to write() in stream " << path);
		TP_LOG_FATAL_ID (strerror (os_errno));
		return NULL;
	    }

	    delete [] tmp_buffer;	// use vector delete -RW
	    file_pointer = os_block_size_;

	    ptr_to_header = new BTE_stream_header;
	    if (ptr_to_header != NULL) {
		return ptr_to_header;
	    } else {
		os_errno = errno;
		TP_LOG_FATAL_ID ("Failed to alloc space for header.");
		TP_LOG_FATAL_ID (strerror (os_errno));
		status_ = BTE_STREAM_STATUS_INVALID;
		return NULL;
	    }
	}
    }

    // Instead of mmap() we simply read in the os_block_size_ leading
    // bytes of the file, copy the leading sizeof(ufs_stream_header)
    // bytes of the os_block_size_ bytes into the ptr_to_header
    // structure and return ptr_to_header. Note that even though we
    // could have read only the first sizeof(ufs_stream_header) of the
    // file we choose not to do so in order to avoid confusing
    // sequential prefetcher.

    char *tmp_buffer = new char[os_block_size_];

    if (TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_SET) != 0) {
	os_errno = errno;
	TP_LOG_FATAL_ID ("Failed to lseek() in stream " << path);
	TP_LOG_FATAL_ID (strerror (os_errno));
	return NULL;
    }

    if (TPIE_OS_READ (fd, (char *) tmp_buffer, os_block_size_) !=
	(TPIE_OS_SSIZE_T) os_block_size_) {
	os_errno = errno;
	TP_LOG_FATAL_ID ("Failed to read() in stream " << path);
	TP_LOG_FATAL_ID (strerror (os_errno));
	return NULL;
    }
    
    file_pointer = os_block_size_;
    ptr_to_header = new BTE_stream_header;
    memcpy(ptr_to_header, tmp_buffer, sizeof(BTE_stream_header));
    delete [] tmp_buffer;	// should use vector delete -RW
    return ptr_to_header;
}

//
// Make sure the current block is mapped in and all internal pointers are
// set as appropriate.  
//  
template < class T >
inline BTE_err BTE_stream_ufs < T >::validate_current (void) {

    unsigned int block_space;	// The space left in the current block.
    BTE_err bte_err;

    // If the current block is valid and current points into it and has
    // enough room in the block for a full item, we are fine.  If it is
    // valid but there is not enough room, unmap it.
    if (block_valid) {
	if ((block_space = (unsigned int)header->block_size -
	     ((char *) current - (char *) curr_block)) >= (unsigned int)sizeof (T)) {
	    return BTE_ERROR_NO_ERROR;
	} else {			// Not enough room left.
	    if ((bte_err = unmap_current ()) != BTE_ERROR_NO_ERROR) {
		return bte_err;
	    }
	    f_offset += block_space;
	}
    }
    // The current block is invalid, since it was either invalid to start
    // with or we just invalidated it because we were out of space.
    tp_assert (!block_valid, "Block is already mapped in.");

    // Now map in the block.
    return map_current ();

}

template < class T >
inline BTE_err BTE_stream_ufs < T >::invalidate_current (void)
{
    // We should currently have a valid block.
    tp_assert (block_valid, "No block is mapped in.");
    block_valid = 0;

    return BTE_ERROR_NO_ERROR;
}

// Map in the current block.
// f_offset is used to determine what block is needed.
template < class T > BTE_err BTE_stream_ufs < T >::map_current (void) {

    TPIE_OS_OFFSET block_offset;
    int do_mmap = 0;

    // We should not currently have a valid block.
    tp_assert (!block_valid, "Block is already mapped in.");

    // Determine the offset of the block that the current item is in.
    block_offset = ((f_offset - os_block_size_) / header->block_size)
	* header->block_size + os_block_size_;

    // If the block offset is beyond the logical end of the file, then
    // we either record this fact and return (if the stream is read
    // only) or ftruncate() out to the end of the current block.
    if (f_filelen < block_offset + (TPIE_OS_OFFSET) header->block_size) {
	if (r_only) {
	    return BTE_ERROR_END_OF_STREAM;
	} else {

	    // An assumption here is that !r_only implies that you won't try to read
	    // items beyond offset block_offset. This is justified because one invariant
	    // being maintained is that file length is os_block_size_ + an INTEGRAL
	    // number of Logical Blocks: By this invariant, since lseek returns something 
	    //  smaller than block_offset + header->block_size - 1 (meaning that filesize
	    // is smaller than block_offset + header->block_size), 
	    //
	    // A consequence of this assumption is that the block being mapped in
	    // is being written/appended. Now while using mmapped I/O, what this 
	    // means is that we need to first ftruncate() and then map in the requisite
	    // block. On the other hand, if we are using the read()/write() BTE, we
	    // simply do nothing: the unmap_current() call executed during
	    // validate_current() and before map_current() would have ensured that
	    // we do not overwrite some previously mapped block. 

	    // Not mapped I/O  
	    // means we assume we are using the read()/write() BTE
	    // This means we do an unmap_current() in validate_current()
	    // just before  map_current() so there's no danger of overwriting
	    // a dirty block.  

	    if (curr_block == NULL) {
		curr_block = new T[(sizeof(T)-1+header->block_size)/sizeof(T)];

		// If you really want to be anal about memory calculation
		// consistency then if BTE_IMPLICIT_FS_READAHEAD flag is
		// set you should register a memory allocation of
		// header->block_size AT THIS POINT of time in code.
	    }

	    block_valid = 1;
	    curr_block_file_offset = block_offset;
	    block_dirty = 0;

	    register TPIE_OS_OFFSET internal_block_offset;

	    internal_block_offset =
		file_off_to_item_off (f_offset) % blocksize_items;

	    current = curr_block + internal_block_offset;

	    return BTE_ERROR_NO_ERROR;

	}
    }
    // If the current block is already mapped in by this process then
    // some systems, (e.g. HP-UX), will not allow us to map it in
    // again.  This presents all kinds of problems, not only with
    // sub/super-stream interactions, which we could probably detect
    // by looking back up the path to the level 0 stream, but also
    // with overlapping substreams, which are very hard to detect
    // since the application can build them however it sees fit.  We
    // can also have problems if we break a stream into two substreams
    // such that their border is in the middle of a block, and then we
    // read to the end of the fisrt substream while we are still at
    // the beginning of the second.

#if UFS_DOUBLE_BUFFER
    if (have_next_block && (block_offset == f_next_block)) {
	T *temp;

	temp = curr_block;
	curr_block = next_block;
	next_block = temp;
	have_next_block = 0;
    } else {
	do_mmap = 1;
    }
#else
    do_mmap = 1;
#endif

    if (do_mmap) {

	if (file_pointer == -1 || block_offset != file_pointer) {
	    if (TPIE_OS_LSEEK(fd, block_offset, TPIE_OS_FLAG_SEEK_SET) != block_offset) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;
		TP_LOG_FATAL_ID ("seek failed in file:");
		TP_LOG_FATAL_ID (path);
		TP_LOG_FATAL_ID(strerror(os_errno));
		return BTE_ERROR_OS_ERROR;
	    }
	}

	if (curr_block == NULL) {
	
	    curr_block = new T[(sizeof(T)-1+header->block_size)/sizeof(T)];

	    // If you really want to be anal about memory calculation
	    // consistency then if BTE_IMPLICIT_FS_READAHEAD flag is set
	    // you shd register a memory allocation of header->block_size
	    // AT THIS POINT of time in code.
	}

	if (TPIE_OS_READ (fd, (char *) curr_block, header->block_size) !=
	    (TPIE_OS_SSIZE_T) header->block_size) {
	    status_ = BTE_STREAM_STATUS_INVALID;
	    os_errno = errno;
	   TP_LOG_FATAL_ID ("read failed in file ");
	   TP_LOG_FATAL_ID(path);
	   TP_LOG_FATAL_ID(strerror(os_errno));
	    return BTE_ERROR_OS_ERROR;
	}

	// Advance file pointer.
	file_pointer = block_offset + header->block_size;
    }
   

    block_valid = 1;
    curr_block_file_offset = block_offset;
    block_dirty = 0;

#if BTE_STREAM_UFS_READ_AHEAD
    // Start the asyncronous read of the next logical block.
    read_ahead ();
#endif

    // The offset, in terms of number of items, that current should
    // have relative to curr_block.
    register TPIE_OS_OFFSET internal_block_offset;

    internal_block_offset =
	file_off_to_item_off (f_offset) % blocksize_items;

    current = curr_block + internal_block_offset;

    gstats_.record(BLOCK_READ);
    stats_.record(BLOCK_READ);
    return BTE_ERROR_NO_ERROR;
}

template < class T > 
BTE_err BTE_stream_ufs < T >::unmap_current (void) {

    ///   TPIE_OS_OFFSET lseek_retval;

    // We should currently have a valid block.
    tp_assert (block_valid, "No block is mapped in.");

    if (!r_only && block_dirty) {

	if (file_pointer == -1 || curr_block_file_offset != file_pointer) {
	    if (TPIE_OS_LSEEK(fd, curr_block_file_offset, TPIE_OS_FLAG_SEEK_SET) !=
		curr_block_file_offset) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;
		TP_LOG_FATAL_ID ("lseek() failed while unmapping current block.");
		TP_LOG_FATAL_ID (strerror(os_errno));
		return BTE_ERROR_OS_ERROR;
	    }
	}

	if (curr_block_file_offset == f_filelen)
	    f_filelen += header->block_size;
     TPIE_OS_OFFSET bla = 0;
	if ((bla = TPIE_OS_WRITE (fd, (char *) curr_block, header->block_size)) !=
	     header->block_size) {
	    status_ = BTE_STREAM_STATUS_INVALID;
	    os_errno = errno;
	   TP_LOG_FATAL_ID ("write() failed to unmap current block.");
	   TP_LOG_FATAL_ID (bla);
	   TP_LOG_FATAL_ID ((TPIE_OS_OFFSET)header->block_size);
	   TP_LOG_FATAL_ID (strerror(os_errno));
	    return BTE_ERROR_OS_ERROR;
	}
	// Advance file pointer.
	file_pointer = curr_block_file_offset + header->block_size;
    }
   
    block_dirty = 0;
    block_valid = 0;
    curr_block_file_offset = 0;

    gstats_.record(BLOCK_WRITE);
    stats_.record(BLOCK_WRITE);
    return BTE_ERROR_NO_ERROR;
}

// A uniform method for advancing the current pointer.  No mapping,
// unmapping, or anything like that is done here.
template < class T >
inline BTE_err BTE_stream_ufs < T >::advance_current (void) {

    // Advance the current pointer and the file offset of the current
    // item.
    current++;
    f_offset += sizeof (T);

    return BTE_ERROR_NO_ERROR;
}

template < class T >
inline TPIE_OS_OFFSET BTE_stream_ufs < T >::item_off_to_file_off (TPIE_OS_OFFSET item_off) const
{
    TPIE_OS_OFFSET file_off;

    if (!itemsize_div_blocksize) {

	// Move past the header.  
	file_off = os_block_size_;

	// Add header->block_size for each full block.
	file_off += header->block_size * (item_off / blocksize_items);

	// Add sizeof(T) for each item in the partially full block.
	file_off += sizeof (T) * (item_off % blocksize_items);

	return file_off;

    } else {

	return (os_block_size_ + item_off * sizeof (T));

    }
}

template < class T >
inline TPIE_OS_OFFSET BTE_stream_ufs < T >::file_off_to_item_off (TPIE_OS_OFFSET file_off) const
{
    TPIE_OS_OFFSET item_off;

    if (!itemsize_div_blocksize) {

	// Subtract off the header.
	file_off -= os_block_size_;

	// Account for the full blocks.
	item_off = blocksize_items * (file_off / header->block_size);

	// Add in the number of items in the last block.
	item_off += (file_off % header->block_size) / sizeof (T);

	return item_off;

    } else {

	return (file_off - os_block_size_) / sizeof (T);

    }
}

template < class T > 
TPIE_OS_OFFSET BTE_stream_ufs < T >::chunk_size (void) const {
    return blocksize_items;
}


#if BTE_STREAM_UFS_READ_AHEAD

template < class T > void BTE_stream_ufs < T >::read_ahead (void)
{
    TPIE_OS_OFFSET f_curr_block;

    // The current block had better already be valid or we made a
    // mistake in being here.
    tp_assert (block_valid,
	       "Trying to read ahead when current block is invalid.");

    // Check whether there is a next block.  If we are already in the
    // last block of the file then it makes no sense to read ahead.
    f_curr_block = ((f_offset - os_block_size_) / header->block_size) *
	header->block_size + os_block_size_;

    if (f_eos < f_curr_block + header->block_size) {
	return;
    }

    f_next_block = f_curr_block + header->block_size;

#if USE_LIBAIO
    // Asyncronously read the first word of each os block in the next
    // logical block.

    for (unsigned int ii = 0; ii < BTE_STREAM_UFS_BLOCK_FACTOR; ii++) {

	// Make sure there is not a pending request for this block
	// before requesting it.  
	if (aio_results[ii].aio_return != AIO_INPROGRESS) {
	    aio_results[ii].aio_return = AIO_INPROGRESS;

	    // We have to cancel the last one, even though it completed,
	    // in order to allow another one with the same result.
	    aiocancel (aio_results + ii);

	    // Start the async I/O.
	    if (::aioread (fd, (char *) (read_ahead_buffer + ii), sizeof (int),
			   f_next_block + ii * os_block_size_, TPIE_OS_FLAG_SEEK_SET,
			   aio_results + ii)) {

		os_errno = errno;
		TP_LOG_FATAL_ID ("aioread() failed to read ahead");
		TP_LOG_FATAL_ID (strerror (os_errno));
	    }
	}
    }

#endif

#if UFS_DOUBLE_BUFFER
#error Explicit double buffering not supported using read/write BTE
#endif
}

#endif	/* BTE_STREAM_UFS_READ_AHEAD */

#undef BTE_STREAM_UFS_MM_BUFFERS

#endif // _BTE_STREAM_UFS_H
