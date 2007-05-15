//
// File: bte_stream_mmap.h (formerly bte_mmb.h)
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/13/94
//
// $Id: bte_stream_mmap.h,v 1.16 2005/07/07 20:36:12 adanner Exp $
//
// Memory mapped streams.  This particular implementation explicitly manages
// blocks, and only ever maps in one block at a time.
//
// TODO: Get rid of or fix the LIBAIO stuff. As it is now it has no
// chance of working, since it uses the static
// BTE_STREAM_MMAP_BLOCK_FACTOR, which is no longer the true
// factor. The true block factor is determined dynamically, from the
// header.
//

#ifndef _BTE_STREAM_MMAP_H
#define _BTE_STREAM_MMAP_H

#ifdef VERBOSE
#  include <iostream>
#endif

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

// For header's type field (77 == 'M').
#define BTE_STREAM_MMAP 77

#include <assert.h>

#if USE_LIBAIO
#  if !HAVE_LIBAIO
#    error USE_LIBAIO requested, but aio library not in configuration.
#  endif
#  include <sys/asynch.h>
#endif

#ifdef BTE_STREAM_MMAP_READ_AHEAD
#  define BTE_STREAM_MMAP_MM_BUFFERS 2
#else
#  define BTE_STREAM_MMAP_MM_BUFFERS 1
#endif

// Get the BTE_stream_base class and other definitions.
#include <bte_stream_base.h>

// This code makes assertions and logs errors.
#include <tpie_assert.h>
#include <tpie_log.h>

#ifndef  BTE_STREAM_MMAP_BLOCK_FACTOR
#  define BTE_STREAM_MMAP_BLOCK_FACTOR 8
#endif

// Figure out the block offset for an offset (pos) in file.
// os_block_size_ is assumed to be the header size.
#define BLOCK_OFFSET(pos) (((pos - os_block_size_) / header->block_size) * header->block_size + os_block_size_)


//
// BTE_stream_mmap<T>
//
// This is a class template for the mmap() based implementation of a 
// BTE stream of objects of type T.  This version maps in only one
// block of the file at a time. 
//
template < class T > class BTE_stream_mmap: public BTE_stream_base < T > {
private:
	unsigned int mmap_status;

	TPIE_OS_FILE_DESCRIPTOR fd;	// descriptor of the mapped file.


	size_t os_block_size_;

	// Offset of the current item in the file.  This is the logical
	// offset of the item within the file, that is, the place we would
	// have to lseek() to in order to read() or write() the item if we
	// were using ordinary (i.e. non-mmap()) file access methods.
	TPIE_OS_OFFSET f_offset;

	// Offset just past the end of the last item in the stream.  If this
	// is a substream, we can't write here or anywhere beyond.
	TPIE_OS_OFFSET f_eos;

	// Length of the file in the file system.  this is the first offset
	// that would be not part of the file. Different from f_eos since
	// we can grow the file independently of the actual writes.
	TPIE_OS_OFFSET f_filelen;

	// Beginning of the file.  Can't write before here.
	TPIE_OS_OFFSET f_bos;

	// A pointer to the mapped in header block for the stream.
	BTE_stream_header *header;

	// Pointer to the current item (mapped in).
	T *current;
	// Pointer to beginning of the currently mapped block.
	T *curr_block;
	// Non-zero if current points to a valid, mapped block.
	int block_valid;
	// true if the curr_block is mapped.
	int block_mapped;

	// for use in double buffering
	T *next_block;		// ptr to next block
	TPIE_OS_OFFSET f_next_block;		// position of next block
	int have_next_block;		// is next block mapped
	int w_only;			// stream is write-only

	// A place to cache OS error values. It is normally set after each
	// call to the OS.
	int os_errno;

	char path[BTE_STREAM_PATH_NAME_LEN];

#if USE_LIBAIO
	// A buffer to read the first word of each OS block in the next logical
	// block for read ahead.
	int read_ahead_buffer[BTE_STREAM_MMAP_BLOCK_FACTOR];

	// Results of asyncronous I/O.
	aio_result_t aio_results[BTE_STREAM_MMAP_BLOCK_FACTOR];
#endif

#ifdef BTE_STREAM_MMAP_READ_AHEAD
	// Read ahead into the next logical block.
	void read_ahead ();
#endif

	void initialize ();

	BTE_stream_header *map_header ();
	void unmap_header ();

	inline BTE_err validate_current ();
	BTE_err map_current ();
	inline BTE_err invalidate_current ();
	BTE_err unmap_current ();

	inline BTE_err advance_current ();

	inline TPIE_OS_OFFSET item_off_to_file_off (TPIE_OS_OFFSET item_off) const;
	inline TPIE_OS_OFFSET file_off_to_item_off (TPIE_OS_OFFSET item_off) const;

#ifdef COLLECT_STATS
	long stats_hits;
	long stats_misses;
	long stats_compulsory;
	long stats_eos;
#endif

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
	using BTE_stream_base<T>::register_memory_allocation;	
	using BTE_stream_base<T>::register_memory_deallocation;

 public:
	// Constructor.
	// [tavi 01/09/02] Careful with the lbf (logical block factor)
	// parameter. I introduced it in order to avoid errors when reading
	// a stream having a different block factor from the default, but
	// this may cause errors in applications. For example,
	// AMI_partition_and merge computes memory requirements of temporary
	// streams based on the memory usage of the INPUT stream. However,
	// the input stream may have different block size from the temporary
	// streams created later. Until these issues are addressed, the
	// usage of lbf is discouraged.
	BTE_stream_mmap (const char *dev_path, BTE_stream_type st, 
		size_t lbf = BTE_STREAM_MMAP_BLOCK_FACTOR);

	// A substream constructor.
	BTE_stream_mmap (BTE_stream_mmap * super_stream,
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
	TPIE_OS_OFFSET tell() const;

	// Truncate the stream.
	BTE_err truncate (TPIE_OS_OFFSET offset);

	// Destructor
	~BTE_stream_mmap ();

	B_INLINE BTE_err read_item (T ** elt);
	B_INLINE BTE_err write_item (const T & elt);

	TPIE_OS_OFFSET chunk_size () const;

	void print (char *pref = "");
	inline BTE_err grow_file (TPIE_OS_OFFSET block_offset);

	static void log_fatal (char *a, char *b, char *c, char *d, char *e) {
		TP_LOG_FATAL (a);
		TP_LOG_FATAL (b);
		TP_LOG_FATAL (c);
		TP_LOG_FATAL (d);
		TP_LOG_FATAL (e);
		TP_LOG_FLUSH_LOG;
	};
};

/* ********************************************************************** */

/* definitions start here */

static int call_munmap (void *addr, size_t len)
{

	//    int rv;
	//
	//#ifdef MACH_ALPHA
	//    rv = TPIE_OS_MUNMAP (addr, len);
	//#else
	//    rv = TPIE_OS_MUNMAP ((caddr_t) addr, len);
	//#endif
	//    return rv;

	return TPIE_OS_MUNMAP (addr, len);
}

static void *call_mmap (void *addr, size_t len, int r_only, int w_only,
						TPIE_OS_FILE_DESCRIPTOR fd, TPIE_OS_OFFSET off, int fixed)
{
	void *ptr;
	int flags = 0;

	assert (!fixed || addr);

#ifdef MACH_ALPHA
	// for enhanced mmap calls
#define MAP_OVERWRITE 0x1000	// block will be overwritten
	flags = (MAP_FILE |
		(fixed ? TPIE_OS_FLAG_MAP_FIXED :MAP_VARIABLE) |
		(w_only ? MAP_OVERWRITE : 0));
#else
	flags = (fixed ? TPIE_OS_FLAG_MAP_FIXED : 0);
#endif
	flags |= TPIE_OS_FLAG_MAP_SHARED;

	//#ifdef MACH_ALPHA
	//    ptr = TPIE_OS_MMAP (addr, len,
	//			(r_only ? PROT_READ : PROT_READ | PROT_WRITE),
	//			flags, fd, off);
	//#else
	//    ptr = TPIE_OS_MMAP ((caddr_t) addr, len,
	//			(r_only ? PROT_READ : PROT_READ | PROT_WRITE),
	//			flags, fd, off);
	//#endif

	ptr = TPIE_OS_MMAP (addr, len,
		(r_only ? TPIE_OS_FLAG_PROT_READ : TPIE_OS_FLAG_PROT_READ | TPIE_OS_FLAG_PROT_WRITE),
		flags, fd, off);
	assert (ptr);
	return ptr;
}


template < class T > void BTE_stream_mmap < T >::initialize ()
{
#ifdef COLLECT_STATS
	stats_misses = stats_hits = stats_compulsory = stats_eos = 0;
#endif
	have_next_block = 0;
	block_valid = 0;
	block_mapped = 0;
	f_offset = f_bos = os_block_size_;
	next_block = curr_block = current = NULL;
	///f_stats = 1;
}

//
// This constructor creates a stream whose contents are taken from the
// file whose path is given.
//
template < class T >
BTE_stream_mmap < T >::BTE_stream_mmap (const char *dev_path, BTE_stream_type st, size_t lbf) {
	status_ = BTE_STREAM_STATUS_NO_STATUS;

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
	r_only = (st == BTE_READ_STREAM);
	w_only = (st == BTE_WRITEONLY_STREAM);

	os_block_size_ = os_block_size();

	// This is a top level stream
	substream_level = 0;
	// Reduce the number of streams available.
	remaining_streams--;

	switch (st) {
	case BTE_READ_STREAM:
		// Open the file for reading.
		if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(fd = TPIE_OS_OPEN_ORDONLY(path, TPIE_OS_FLAG_USE_MAPPING_TRUE))) {
			status_ = BTE_STREAM_STATUS_INVALID;
			os_errno = errno;
			log_fatal ("open() failed to open \"", path, "\": ",
				strerror (os_errno), "\n");
			// [tavi 01/07/02] Commented this out. No need to panic.
			//assert (0);
			return;
		}
		// Get ready to read the first item out of the file.
		initialize ();
		header = map_header ();
		if (check_header (header) < 0) {
			status_ = BTE_STREAM_STATUS_INVALID;
			// [tavi 01/07/02] Commented this out. No need to panic.
			//assert (0);
			return;
		}
		if (header->type != BTE_STREAM_MMAP) {
			TP_LOG_WARNING_ID("Using MMAP stream implem. on another type of stream.");
			TP_LOG_WARNING_ID("Stream implementations may not be compatible.");
		}
		if ((header->block_size % os_block_size_ != 0) || 
			(header->block_size == 0)) {
				status_ = BTE_STREAM_STATUS_INVALID;
				TP_LOG_FATAL_ID ("header: incorrect logical block size;");
				TP_LOG_FATAL_ID ("expected multiple of OS block size.");
				return;
			}
			if (header->block_size != BTE_STREAM_MMAP_BLOCK_FACTOR * os_block_size_) {
				TP_LOG_WARNING_ID("Stream has different block factor than the default.");
				TP_LOG_WARNING_ID("This may cause problems in some existing applications.");
			}
			break;

	case BTE_WRITE_STREAM:
	case BTE_WRITEONLY_STREAM:
	case BTE_APPEND_STREAM:
		// Open the file for writing.  First we will try to open 
		// is with the O_EXCL flag set.  This will fail if the file
		// already exists.  If this is the case, we will call open()
		// again without it and read in the header block.
		if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(fd = TPIE_OS_OPEN_OEXCL(path, TPIE_OS_FLAG_USE_MAPPING_TRUE))) {

			// Try again, hoping the file already exists.
			if (!TPIE_OS_IS_VALID_FILE_DESCRIPTOR(fd = TPIE_OS_OPEN_ORDWR(path, TPIE_OS_FLAG_USE_MAPPING_TRUE))) {
				status_ = BTE_STREAM_STATUS_INVALID;
				os_errno = errno;
				log_fatal ("open() failed to open \"", path, "\": ",
					strerror (os_errno), "\n");
				return;
			}
			initialize ();
			// The file already exists, so read the header.
			header = map_header ();
			if (check_header (header) < 0) {
				status_ = BTE_STREAM_STATUS_INVALID;
				// [tavi 01/07/02] Commented this out. No need to panic.
				//assert (0);
				return;
			}
			if (header->type != BTE_STREAM_MMAP) {
				TP_LOG_WARNING_ID("Using MMAP stream implem. on another type of stream.");
				TP_LOG_WARNING_ID("Stream implementations may not be compatible.");
			}
			if ((header->block_size % os_block_size_ != 0) || 
				(header->block_size == 0)) {
					status_ = BTE_STREAM_STATUS_INVALID;
					TP_LOG_FATAL_ID ("header: incorrect logical block size;");
					TP_LOG_FATAL_ID ("expected multiple of OS block size.");
					return;
				}
				if (header->block_size != BTE_STREAM_MMAP_BLOCK_FACTOR * os_block_size_) {
					TP_LOG_WARNING_ID("Stream has different block factor than the default.");
					TP_LOG_WARNING_ID("This may cause problems in some existing applications.");
				}
		} else {	 // The file was just created.

			f_eos = os_block_size_;
			// Rajiv
			// [tavi 01/07/02] Commented this out. Aren't we sure the file is OK?
			//assert (lseek (fd, 0, SEEK_END) == 0);

#ifdef VERBOSE
			if (verbose)
				cout << "CONS created file: " << path << "\n";
#endif

			// what does this do??? Rajiv
			// Create and map in the header.
			if (TPIE_OS_LSEEK(fd, os_block_size_ - 1, TPIE_OS_FLAG_SEEK_SET) != os_block_size_ - 1) {
				status_ = BTE_STREAM_STATUS_INVALID;
				os_errno = errno;
				log_fatal ("lseek() failed to move past header of \"",
					path, "\": ", strerror (os_errno), "\n");
				// [tavi 01/07/02] Commented this out. No need to panic.
				//assert (0 == 1);
				return;
			}
			initialize ();
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
			header->type = BTE_STREAM_MMAP;
			gstats_.record(STREAM_CREATE);
			stats_.record(STREAM_CREATE);
		}
		break;
	}

	// We can't handle streams of large objects.
	if (sizeof (T) > header->block_size) {
		status_ = BTE_STREAM_STATUS_INVALID;
		TP_LOG_FATAL_ID ("Object is too big (object size/block size):");
		TP_LOG_FATAL_ID (sizeof(T));
		TP_LOG_FATAL_ID (static_cast<TPIE_OS_OUTPUT_SIZE_T>(header->block_size));
		return;
	}
	f_filelen = TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_END);
	assert (f_filelen >= 0);
	f_eos = item_off_to_file_off (header->item_logical_eof);
	if (st == BTE_APPEND_STREAM) {
		f_offset = f_eos;
	} else {
		f_offset = os_block_size_;
	}
#ifdef VERBOSE
	// Rajiv
	if (verbose)
		cout << "CONS logical eof=" << header->item_logical_eof << "\n";
#endif

	// By default, all streams are deleted at destruction time.
	// [tavi 01/07/02] No. Streams initialized with given names are persistent.
	per = PERSIST_PERSISTENT;

	// Register memory usage before returning.
  // Since blocks and header are allocated by mmap and not "new",
  // register memory manually. No mem overhead with mmap
	register_memory_allocation (sizeof (BTE_stream_header));
	register_memory_allocation (BTE_STREAM_MMAP_MM_BUFFERS * header->block_size);
	gstats_.record(STREAM_OPEN);
	stats_.record(STREAM_OPEN);

#ifdef VERBOSE
	// Rajiv
	if (verbose) {
		switch (st) {
	case BTE_READ_STREAM:
		cout << "CONS read stream\n";
		break;
	case BTE_WRITE_STREAM:
		cout << "CONS read/write stream\n";
		break;
	default:
		cout << "CONS someother stream\n";
		break;
		}
		print ("CONS ");
	}
#endif
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
BTE_stream_mmap < T >::BTE_stream_mmap (BTE_stream_mmap * super_stream, BTE_stream_type st, TPIE_OS_OFFSET sub_begin, TPIE_OS_OFFSET sub_end) {

	status_ = BTE_STREAM_STATUS_NO_STATUS;

	if (remaining_streams <= 0) {
		TP_LOG_FATAL_ID ("BTE error: cannot open more streams.");
		status_ = BTE_STREAM_STATUS_INVALID;
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
	// Rajiv
	initialize ();

	// Reduce the number of streams avaialble.
	remaining_streams--;
	// Copy the relevant fields from the super_stream.
	fd = super_stream->fd;
	os_block_size_ = super_stream->os_block_size_;
	header = super_stream->header;
	f_filelen = super_stream->f_filelen;
	substream_level = super_stream->substream_level + 1;

	per = PERSIST_PERSISTENT;

	// The arguments sub_start and sub_end are logical item positions
	// within the stream.  We need to convert them to offsets within
	// the stream where items are found.

	TPIE_OS_OFFSET super_item_begin = file_off_to_item_off (super_stream->f_bos);

	f_bos = item_off_to_file_off (super_item_begin + sub_begin);
	f_eos = item_off_to_file_off (super_item_begin + sub_end + 1);

	if (f_eos > super_stream->f_eos) {
		status_ = BTE_STREAM_STATUS_INVALID;
		return;
	}

	f_offset = f_bos;

	curr_block = NULL;
	block_valid = 0;

	r_only = super_stream->r_only;
	w_only = super_stream->w_only;

	strncpy (path, super_stream->path, BTE_STREAM_PATH_NAME_LEN);
	gstats_.record(STREAM_OPEN);
	gstats_.record(SUBSTREAM_CREATE);
	stats_.record(STREAM_OPEN);
	stats_.record(SUBSTREAM_CREATE);

	// substreams are considered to have no memory overhead!
}

// A psuedo-constructor for substreams.  This serves as a wrapper for
// the constructor above in order to get around the fact that one
// cannot have virtual constructors.
template < class T >
BTE_err BTE_stream_mmap < T >::new_substream (BTE_stream_type st, TPIE_OS_OFFSET sub_begin, TPIE_OS_OFFSET sub_end, BTE_stream_base < T > **sub_stream) {
	// Check permissions.
	if ((st != BTE_READ_STREAM) && ((st != BTE_WRITE_STREAM) || r_only)) {
		*sub_stream = NULL;
		return BTE_ERROR_PERMISSION_DENIED;
	}

	tp_assert (((st == BTE_WRITE_STREAM) && !r_only) ||
		(st == BTE_READ_STREAM),
		"Bad things got through the permisssion checks.");

	BTE_stream_mmap < T > *sub =
		new BTE_stream_mmap < T > (this, st, sub_begin, sub_end);

	*sub_stream = (BTE_stream_base < T > *)sub;

	return BTE_ERROR_NO_ERROR;
}

template < class T > BTE_stream_mmap < T >::~BTE_stream_mmap (void) {

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
	// If this is writable and not a substream, then put the logical
	// eos back into the header before unmapping it.
	if (!r_only && !substream_level) {
		header->item_logical_eof = file_off_to_item_off (f_eos);
	}
#ifdef VERBOSE
	// Rajiv
	if (verbose) {
		cout << "DELE logical eof=" << header->item_logical_eof << "\n";
	}
#endif
	assert (substream_level ||
		header->item_logical_eof == file_off_to_item_off (f_eos));

	// Unmap the current block if necessary.
	if (block_mapped) {
		unmap_current();
	}
	// If this is not a substream then close the file.
	if (!substream_level) {
		// [Rajiv] make sure the length of the file is correct
		// [tavi 06/23/02] and file is not read-only! 
		if ((f_filelen > f_eos) && (!r_only) &&
			(TPIE_OS_FTRUNCATE (fd, BLOCK_OFFSET(f_eos) + header->block_size) < 0)) {
				os_errno = errno;
				TP_LOG_FATAL_ID("Failed to ftruncate() to the new end of " << path);
				TP_LOG_FATAL_ID("f_filelen:" << f_filelen << ", f_eos:" << f_eos);
				TP_LOG_FATAL_ID("argument to ftruncate:" << BLOCK_OFFSET(f_eos) + header->block_size);
				TP_LOG_FATAL_ID(strerror (os_errno));
			}

			// Unmap the header.
			// [tavi 06/23/02] Added test for r_only. 
			if (!r_only)
				unmap_header();
			// Close the file.
			if (TPIE_OS_CLOSE (fd)) {
				os_errno = errno;
				TP_LOG_WARNING_ID("Failed to close() " << path);
				TP_LOG_WARNING_ID(strerror (os_errno));
			}
			// If it should not persist, unlink the file.
			if (per == PERSIST_DELETE) {
				if (r_only) {
					TP_LOG_WARNING_ID("PERSIST_DELETE for read-only stream in " << path);
				}
				else {
					if (TPIE_OS_UNLINK (path)) {
						os_errno = errno;
						TP_LOG_WARNING_ID ("unlink() failed during destruction of " << path);
						TP_LOG_WARNING_ID (strerror (os_errno));
					} else {
						gstats_.record(STREAM_DELETE);
						stats_.record(STREAM_DELETE);
					}
				}
			}

			// Register memory deallocation before returning.
			register_memory_deallocation (sizeof (BTE_stream_header)); // for the header.
			register_memory_deallocation (BTE_STREAM_MMAP_MM_BUFFERS *
				header->block_size);

			// Free the in-memory header (previously allocated with malloc).
			free (header);

	} else {
		gstats_.record(SUBSTREAM_DELETE);
		stats_.record(SUBSTREAM_DELETE);
	}

	gstats_.record(STREAM_CLOSE);
	stats_.record(STREAM_CLOSE);

#ifdef VERBOSE
	if (verbose) {
		if (per == PERSIST_DELETE)
			cout << "DELE unlinked file\n";
		if (substream_level) {
			cout << "DELE substream destructor\n";
		}
		print ("DELE ");
	}
#endif
}

// pref = prefix
template < class T > void BTE_stream_mmap < T >::print (char *pref) {

#ifdef COLLECT_STATS
#ifdef BTE_STREAM_MMAP_READ_AHEAD
	fprintf (stdout, "%sPFSTATS %d %d %d %d (fd=%d)\n",
		pref, stats_hits, stats_misses,
		stats_compulsory - stats_misses, stats_eos, fd);
	cout << pref << stats << " RWMUSCD\n";
#endif
#endif
	fprintf (stdout, "%sfile=%s", pref, path);
	fprintf (stdout, ", f_eos=%d, f_filelen=%d", f_eos, f_filelen);
	fprintf (stdout, ", length=%d\n",
		file_off_to_item_off (f_eos) - file_off_to_item_off (f_bos));
	fprintf (stdout, "\n");
}

// f_eos points just past the last item ever written, so if f_current
// is there we are at the end of the stream and cannot read.

template < class T >
B_INLINE BTE_err BTE_stream_mmap < T >::read_item (T ** elt) {

	BTE_err bte_err;

	if (w_only) {
#ifdef VERBOSE
		if (verbose)
			cerr << "ERROR read on a write-only stream\n";
#endif
		return BTE_ERROR_WRITE_ONLY;
	}
	// Make sure we are not currently at the EOS.
	if (f_offset + sizeof (T) > f_eos) {
		//tp_assert(0, "Can't read past eos.");
		//TP_LOG_WARNING("Reading past eos.\n");
		return BTE_ERROR_END_OF_STREAM;
	}
	// Validate the current block.
	if ((bte_err = validate_current ()) != BTE_ERROR_NO_ERROR) {
		return bte_err;
	}
	// Check and make sure that the current pointer points into the current
	// block.
	tp_assert (((char *) current - (char *) curr_block <=
		header->block_size - sizeof (T)),
		"current is past the end of the current block");
	tp_assert (((char *) current - (char *) curr_block >= 0),
		"current is before the begining of the current block");

	gstats_.record(ITEM_READ);
	stats_.record(ITEM_READ);

	*elt = current;		// Read
	advance_current ();		// move ptr to next elt

	// If we are in a substream, there should be no way for f_current to
	// pass f_eos.
	tp_assert (!substream_level || (f_offset <= f_eos),
		"Got past eos in a substream.");

	return BTE_ERROR_NO_ERROR;
}

// f_eos points just past the last item ever written, so if f_current
// is there we are at the end of the stream and can only write if this
// is not a substream.
template < class T >
B_INLINE BTE_err BTE_stream_mmap < T >::write_item (const T & elt) {

	BTE_err bte_err;

	///   if (f_stats)
	///      stats.record_write ();

	// This better be a writable stream.
	if (r_only) {
		TP_LOG_WARNING_ID ("write on a read-only stream\n");
		return BTE_ERROR_READ_ONLY;
	}
	// Make sure we are not currently at the EOS of a substream.
	if (substream_level && (f_eos <= f_offset)) {
		tp_assert (f_eos == f_offset, "Went too far in a substream.");
		return BTE_ERROR_END_OF_STREAM;
	}
	// Validate the current block.
	bte_err = validate_current ();
	if (bte_err != BTE_ERROR_NO_ERROR) {
		return bte_err;
	}
	// Check and make sure that the current pointer points into the current
	// block.
	tp_assert (((char *) current - (char *) curr_block <=
		header->block_size - sizeof (T)),
		"current is past the end of the current block");
	tp_assert (((char *) current - (char *) curr_block >= 0),
		"current is before the begining of the current block");
	assert (current);

	gstats_.record(ITEM_WRITE);
	stats_.record(ITEM_WRITE);

	*current = elt;		// write
	advance_current ();		// Advance the current pointer.

	// If we are in a substream, there should be no way for f_current
	// to pass f_eos.
	tp_assert (!substream_level || (f_offset <= f_eos),
		"Got past eos in a substream.");

	// If we moved past eos, then update eos unless we are in a
	// substream, in which case EOS will be returned on the next call.
	if ((f_offset > f_eos) && !substream_level) {
		// I dont like this assert Rajiv
		// tp_assert(f_offset == f_eos + sizeof(T), "Advanced too far somehow.");
		tp_assert (f_offset <= f_filelen, "Advanced too far somehow.");
		f_eos = f_offset;
		// this is the only place f_eos is changed excluding 
		// constructors and truncate Rajiv
	}

	return BTE_ERROR_NO_ERROR;
}

// Query memory usage

// Note that in a substream we do not charge for the memory used by
// the header, since it is accounted for in the 0 level superstream.
template < class T >
BTE_err BTE_stream_mmap < T >::main_memory_usage (size_t * usage, MM_stream_usage usage_type) {
	switch (usage_type) {
    case MM_STREAM_USAGE_OVERHEAD:
      //Fixed costs. Only 2*mem overhead, because only class and base
      //are allocated dynamicall via "new". Header is read via mmap
      *usage = sizeof(*this) + sizeof(BTE_stream_header) +
        2*MM_manager.space_overhead();
      break;
    case MM_STREAM_USAGE_BUFFER:
      //no mem manager overhead when allocated via mmap
      *usage = BTE_STREAM_MMAP_MM_BUFFERS * header->block_size;
      break;
    case MM_STREAM_USAGE_CURRENT:
      *usage = (sizeof(*this) + sizeof(BTE_stream_header) +
          2*MM_manager.space_overhead() + 
          ((curr_block == NULL) ? 0 :
           BTE_STREAM_MMAP_MM_BUFFERS * header->block_size));
      break;
    case MM_STREAM_USAGE_MAXIMUM:
    case MM_STREAM_USAGE_SUBSTREAM:
      *usage = (sizeof(*this) + sizeof(BTE_stream_header) +
          2*MM_manager.space_overhead() + 
          BTE_STREAM_MMAP_MM_BUFFERS * header->block_size);
      break;
  }

	return BTE_ERROR_NO_ERROR;
};

// Return the number of items in the stream.
template < class T > 
TPIE_OS_OFFSET BTE_stream_mmap < T >::stream_len () const {
	return file_off_to_item_off (f_eos) - file_off_to_item_off (f_bos);
};

// Return the path name in newly allocated space.
template < class T >
BTE_err BTE_stream_mmap < T >::name (char **stream_name)
{
	size_t len = strlen (path);

	tp_assert (len < BTE_STREAM_PATH_NAME_LEN, "Path length is too long.");

	// Return the path name in newly allocated space.

	char *new_path = new char[len + 1];

	strncpy (new_path, path, len + 1);

	*stream_name = new_path;

	return BTE_ERROR_NO_ERROR;
};

// Move to a specific position.
template < class T > 
BTE_err BTE_stream_mmap < T >::seek (TPIE_OS_OFFSET offset) {

	BTE_err be;
	TPIE_OS_OFFSET new_offset;

	// Looks like we can only seek within the file Rajiv
	if ((offset < 0) ||
		(offset >
		file_off_to_item_off (f_eos) - file_off_to_item_off (f_bos))) {
			return BTE_ERROR_OFFSET_OUT_OF_RANGE;
		}
		// Compute the new offset
		new_offset =
			item_off_to_file_off (file_off_to_item_off (f_bos) + offset);
		if (r_only) {
			tp_assert (new_offset <= f_eos, "Advanced too far somehow.");
		}
		// // If it is not in the same block as the current position then
		// // invalidate the current block.
		//if (((new_offset - os_block_size_) / header->block_size) !=
		//    ((f_offset - os_block_size_) / header->block_size)) {

		// The above was the old code which was wrong: we also need to check that
		// we have the correct block mapped in (f_offset does not always point into
		// the current block!)

		if (((char *) current - (char *) curr_block >= header->block_size) ||
			(((new_offset - os_block_size_) / header->block_size) !=
			((f_offset - os_block_size_) / header->block_size))) {
/*				
				if (block_valid) {
				if ((be = invalidate_current ()) != BTE_ERROR_NO_ERROR)
*/
				if (block_valid && ((be = unmap_current()) != BTE_ERROR_NO_ERROR)) {
						return be;
				}
			} else {
				if (block_valid) {

					// We have to adjust current.

					register TPIE_OS_OFFSET internal_block_offset;

					internal_block_offset = file_off_to_item_off (new_offset) %
						(header->block_size / sizeof (T));

					current = curr_block + internal_block_offset;
				}
			}

			f_offset = new_offset;

			gstats_.record(ITEM_SEEK);
			stats_.record(ITEM_SEEK);
			return BTE_ERROR_NO_ERROR;
}

template < class T > 
TPIE_OS_OFFSET BTE_stream_mmap < T >::tell() const {
	return file_off_to_item_off(f_offset);
}

// Truncate the stream.
template < class T > 
BTE_err BTE_stream_mmap < T >::truncate (TPIE_OS_OFFSET offset)
{
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
	new_offset =
		item_off_to_file_off (file_off_to_item_off (f_bos) + offset);

	// If it is not in the same block as the current position then
	// invalidate the current block.

	// We also need to check that we have the correct block mapped in (f_offset
	// does not always point into the current block!) - see comment in seek()

	if (((char *) current - (char *) curr_block >= header->block_size) ||
		(((new_offset - os_block_size_) / header->block_size) !=
		((f_offset - os_block_size_) / header->block_size))) {
/*				
				if (block_valid) {
				if ((be = invalidate_current ()) != BTE_ERROR_NO_ERROR)
*/
				if (block_valid && ((be = unmap_current()) != BTE_ERROR_NO_ERROR)) {
						return be;
			}
		}
		// If it is not in the same block as the current end of stream
		// then truncate the file to the end of the new last block.
		if (((new_offset - os_block_size_) / header->block_size) !=
			((f_eos - os_block_size_) / header->block_size)) {

				if(block_mapped) {	
					unmap_current();
				}

				// Determine the offset of the block that new_offset is in.
				block_offset = BLOCK_OFFSET (new_offset);
				// Rajiv 
				// ((new_offset - os_block_size_) / header->block_size)
				// * header->block_size + os_block_size_;
				f_filelen = block_offset + header->block_size;
				if (TPIE_OS_FTRUNCATE (fd, f_filelen)) {
					os_errno = errno;
					TP_LOG_FATAL ("Failed to ftruncate() to the new end of \"");
					TP_LOG_FATAL (path);
					TP_LOG_FATAL ("\": ");
					TP_LOG_FATAL (strerror (os_errno));
					TP_LOG_FATAL ('\n');
					TP_LOG_FLUSH_LOG;
					return BTE_ERROR_OS_ERROR;
				}
			}
			// Reset the current position to the end.    
			f_offset = f_eos = new_offset;

			return BTE_ERROR_NO_ERROR;
}

// Map in the header from the file.  This assumes that the path
// has been cached in path and that the file has been opened and
// fd contains a valid descriptor.
template < class T >
BTE_stream_header * BTE_stream_mmap < T >::map_header (void) {

	TPIE_OS_OFFSET file_end;
	BTE_stream_header *mmap_hdr;

	// If the underlying file is not at least long enough to contain
	// the header block, then, assuming the stream is writable, we have
	// to create the space on disk by doing an explicit write().
	if ((file_end = TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_END)) < os_block_size_) {
		if (r_only) {
			status_ = BTE_STREAM_STATUS_INVALID;

			TP_LOG_FATAL ("No header block in read only stream \"");
			TP_LOG_FATAL (path);
			TP_LOG_FATAL ('\n');
			TP_LOG_FLUSH_LOG;
			return NULL;

		} else {
			// A writable stream, so we can ftruncate() space for a
			// header block.
			if (TPIE_OS_FTRUNCATE (fd, os_block_size_)) {
				os_errno = errno;
				TP_LOG_FATAL ("Failed to ftruncate() to end of header of \"");
				TP_LOG_FATAL (path);
				TP_LOG_FATAL ("\": ");
				TP_LOG_FATAL (strerror (os_errno));
				TP_LOG_FATAL ('\n');
				TP_LOG_FLUSH_LOG;
				return NULL;
			}
		}
	}

	// Map in the header block.  If the stream is writable, the header 
	// block should be too.
	// took out the SYSTYPE_BSD ifdef for convenience
	// changed from MAP_FIXED to MAP_VARIABLE because we are using NULL
	mmap_hdr = (BTE_stream_header *)
		(call_mmap ((NULL), sizeof (BTE_stream_header),
		r_only, w_only, fd, 0, 0));
	if (mmap_hdr == (BTE_stream_header *) (-1)) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;
		log_fatal ("mmap() failed to map in header from \"",
			path, "\": ", strerror (os_errno), "\n");
		return NULL;
	}

	header = (BTE_stream_header *) malloc (sizeof (BTE_stream_header));
	if (!header) {
		TP_LOG_FATAL ("out of virtual memory");
		return NULL;
	}
	memcpy (header, mmap_hdr, sizeof (BTE_stream_header));
	call_munmap (mmap_hdr, sizeof (BTE_stream_header));

	return header;
}

// Map in the header from the file.  This assumes that the path
// has been cached in path and that the file has been opened and
// fd contains a valid descriptor.
template < class T > void BTE_stream_mmap < T >::unmap_header ()
{
	BTE_stream_header *mmap_hdr;
	TPIE_OS_OFFSET file_end;

	// If the underlying file is not at least long enough to contain
	// the header block, then, assuming the stream is writable, we have
	// to create the space on disk by doing an explicit write().
	if ((file_end = TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_END)) < os_block_size_) {
		if (r_only) {
			status_ = BTE_STREAM_STATUS_INVALID;

			TP_LOG_FATAL ("No header block in read only stream \"");
			TP_LOG_FATAL (path);
			TP_LOG_FATAL ('\n');
			TP_LOG_FLUSH_LOG;
			return;

		} else {
			// A writable stream, so we can ftruncate() space for a
			// header block.
			if (TPIE_OS_FTRUNCATE (fd, os_block_size_)) {
				os_errno = errno;
				TP_LOG_FATAL ("Failed to ftruncate() to end of header of \"");
				TP_LOG_FATAL (path);
				TP_LOG_FATAL ("\": ");
				TP_LOG_FATAL (strerror (os_errno));
				TP_LOG_FATAL ('\n');
				TP_LOG_FLUSH_LOG;
				return;
			}
		}
	}

	// Map in the header block.  If the stream is writable, the header 
	// block should be too.
	// took out the SYSTYPE_BSD ifdef for convenience
	// changed from MAP_FIXED to MAP_VARIABLE because we are using NULL
	mmap_hdr = (BTE_stream_header *)
		(call_mmap ((NULL), sizeof (BTE_stream_header),
		r_only, w_only, fd, 0, 0));
	if (mmap_hdr == (BTE_stream_header *) (-1)) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;
		log_fatal ("mmap() failed to map in header from \"",
			path, "\": ", strerror (os_errno), "\n");
		return;
	}

	memcpy (mmap_hdr, header, sizeof (BTE_stream_header));
	call_munmap (mmap_hdr, sizeof (BTE_stream_header));
}

//
// Make sure the current block is mapped in and all internal pointers are
// set as appropriate.  
// 
// 

template < class T >
inline BTE_err BTE_stream_mmap < T >::validate_current (void)
{
	TPIE_OS_SIZE_T block_space;		// The space left in the current block.
	BTE_err bte_err;

	// If the current block is valid and current points into it and has
	// enough room in the block for a full item, we are fine.  If it is
	// valid but there is not enough room, invalidate it.
	if (block_valid) {
		assert (current);		// sanity check - rajiv
		if ((block_space = header->block_size -
			((char *) current - (char *) curr_block)) >= sizeof (T)) {
				return BTE_ERROR_NO_ERROR;
			} else {			// Not enough room left.
				// no real need to call invalidate here 
				// since we call map_current anyway Rajiv
				if ((bte_err = unmap_current ()) != BTE_ERROR_NO_ERROR) {
					return bte_err;
				}
				f_offset += block_space;
			}
	}
	// The current block is invalid, since it was either invalid to start
	// with or we just invalidated it because we were out of space.

	tp_assert (!block_valid, "Block is already mapped in.");

	// Now map it the block.
	bte_err = map_current ();
	assert (current);

#ifdef VERBOSE
	// Rajiv
	if (verbose && bte_err != BTE_ERROR_NO_ERROR)
		cerr << "validate_current failed\n";
#endif

	// Rajiv
	tp_assert (f_offset + sizeof (T) <= f_filelen,
		"Advanced too far somehow.");
	return bte_err;
}

// Map in the current block.
// f_offset is used to determine what block is needed.
template < class T > BTE_err BTE_stream_mmap < T >::map_current (void)
{
	TPIE_OS_OFFSET block_offset;
	int do_mmap = 0;
	BTE_err err;

	// We should not currently have a valid block.
	tp_assert (!block_valid, "Block is already mapped in.");

	// Determine the offset of the block that the current item is in.
	block_offset = BLOCK_OFFSET (f_offset);
	// Rajiv
	// - os_block_size_) / header->block_size)
	// * header->block_size + os_block_size_;

	// If the block offset is beyond the logical end of the file, then
	// we either record this fact and return (if the stream is read
	// only) or ftruncate() out to the end of the current block.
	assert (TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_END) == f_filelen);
	// removed -1 from rhs of comparison below Rajiv
	if (f_filelen < block_offset + header->block_size) {
		if (r_only) {
			//TP_LOG_WARNING_ID("hit eof while reading\n");
			return BTE_ERROR_END_OF_STREAM;
		} else {
			err = grow_file (block_offset);
			if (err != BTE_ERROR_NO_ERROR)
				return err;
		}
	}
	// this is what we just fixed. Rajiv
	tp_assert (f_offset + sizeof (T) <= f_filelen,
		"Advanced too far somehow.");

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

	// Map it in either r/w or read only.  
#ifdef BTE_STREAM_MMAP_READ_AHEAD
	if (have_next_block && (block_offset == f_next_block)) {
		T *temp;

		temp = curr_block;
		curr_block = next_block;
		next_block = temp;
		have_next_block = 0;
#ifdef COLLECT_STATS
		stats_hits++;
#endif
	} else {
#ifdef COLLECT_STATS
		if (have_next_block) {
			// not sequential access
			//munmap((caddr_t)next_block, header->block_size);
			//have_next_block = 0;
			//next_block = NULL;
			stats_misses++;
		}
		stats_compulsory++;
#endif
		do_mmap = 1;
	}
#else
	do_mmap = 1;
#endif
	if (do_mmap) {
		// took out the SYSTYPE_BSD ifdef for convenience
		// MAP_VARIABLE the first time round
		// (curr_block ? MAP_FIXED : MAP_VARIABLE) |
		if (block_offset + header->block_size > f_filelen) {
			grow_file(block_offset);
		}
		curr_block = (T *) (call_mmap (curr_block, header->block_size,
			r_only, w_only, fd, block_offset,
			(curr_block != NULL)));
		block_mapped = 1;
	}
	assert ((void *) curr_block != (void *) header);

	if (curr_block == (T *) (-1)) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;
		TP_LOG_FATAL ("mmap() failed to map in block at ");
		TP_LOG_FATAL (block_offset);
		TP_LOG_FATAL (" from \"");
		TP_LOG_FATAL (path);
		TP_LOG_FATAL ("\": ");
		TP_LOG_FATAL (strerror (os_errno));
		TP_LOG_FATAL ('\n');
		TP_LOG_FLUSH_LOG;
		perror ("mmap failed");	// Rajiv
		return BTE_ERROR_OS_ERROR;
	}
	block_valid = 1;    

#ifdef BTE_STREAM_MMAP_READ_AHEAD
	// Start the asyncronous read of the next logical block.
	read_ahead ();
#endif

	// The offset, in terms of number of items, that current should
	// have relative to curr_block.

	register TPIE_OS_OFFSET internal_block_offset;

	internal_block_offset = file_off_to_item_off (f_offset) %
		(header->block_size / sizeof (T));

	current = curr_block + internal_block_offset;
	assert (current);

	gstats_.record(BLOCK_READ);
	stats_.record(BLOCK_READ);
	return BTE_ERROR_NO_ERROR;
}

template < class T >
inline BTE_err BTE_stream_mmap < T >::invalidate_current (void)
{
	// We should currently have a valid block.
	tp_assert (block_valid, "No block is mapped in.");
	block_valid = 0;

	return BTE_ERROR_NO_ERROR;
}

template < class T > BTE_err BTE_stream_mmap < T >::unmap_current (void) {

	//  2003/02/27: Commented this out as it causes test_ami_pmerge to crash.
	//  (reason: destructor for superstream fails when using substreams)
	//  Jan.
	//    invalidate_current ();	// not really necessary

	// Unmap it.
	if (call_munmap (curr_block, header->block_size)) {
		status_ = BTE_STREAM_STATUS_INVALID;
		os_errno = errno;

		TP_LOG_FATAL ("munmap() failed to unmap current block");
		TP_LOG_FATAL ("\": ");
		TP_LOG_FATAL (strerror (os_errno));
		TP_LOG_FATAL ('\n');
		TP_LOG_FLUSH_LOG;
		return BTE_ERROR_OS_ERROR;
	}
	curr_block = NULL;		// to be safe
	block_mapped = 0;
	block_valid = 0;

	gstats_.record(BLOCK_WRITE);
	stats_.record(BLOCK_WRITE);
	return BTE_ERROR_NO_ERROR;
}

// A uniform method for advancing the current pointer.  No mapping,
// unmapping, or anything like that is done here.
template < class T >
inline BTE_err BTE_stream_mmap < T >::advance_current (void) {

	tp_assert (f_offset <= f_filelen, "Advanced too far somehow.");

	// Advance the current pointer and the file offset of the current
	// item.
	current++;
	f_offset += sizeof (T);

	return BTE_ERROR_NO_ERROR;
}


// increase the length of the file, to at least 
// block_offset + header->block_size
template < class T > inline BTE_err
BTE_stream_mmap < T >::grow_file (TPIE_OS_OFFSET block_offset)
{
	// can't grow substreams (except if called for the
	// last substream in a stream. this may happen if map_current
	// maps in the last block of a (sub-)stream).
	// (tavi) I took this out since ignoreSubstream is not declared...
	//    assert (ignoreSubstream || !substream_level);
	assert(!substream_level);

	f_filelen = block_offset + header->block_size;
	if (TPIE_OS_FTRUNCATE (fd, f_filelen) < 0) {
		os_errno = errno;
		TP_LOG_FATAL ("Failed to ftruncate() out a new block of \"");
		TP_LOG_FATAL (path);
		TP_LOG_FATAL ("\": ");
		TP_LOG_FATAL (strerror (os_errno));
		TP_LOG_FATAL ('\n');
		TP_LOG_FLUSH_LOG;
		// Rajiv
		//cerr << "map_current: ftruncate\n";
		return BTE_ERROR_END_OF_STREAM;	// generate an error Rajiv
	}
	assert (TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_END) == f_filelen);
	return BTE_ERROR_NO_ERROR;
}


template < class T >
TPIE_OS_OFFSET BTE_stream_mmap < T >::item_off_to_file_off (TPIE_OS_OFFSET item_off) const {
	TPIE_OS_OFFSET file_off;

	// Move past the header.

	file_off = os_block_size_;

	// Add header->block_size for each full block.

	file_off += header->block_size *
		(item_off / (header->block_size / sizeof (T)));

	// Add sizeof(T) for each item in the partially full block.

	file_off += sizeof (T) * (item_off % (header->block_size / sizeof (T)));

	return file_off;
}

template < class T >
TPIE_OS_OFFSET BTE_stream_mmap < T >::file_off_to_item_off (TPIE_OS_OFFSET file_off) const {
	TPIE_OS_OFFSET item_off;

	// Subtract off the header.
	file_off -= os_block_size_;

	// Account for the full blocks.
	item_off = (header->block_size / sizeof (T)) *
		(file_off / header->block_size);

	// Add in the number of items in the last block.
	item_off += (file_off % header->block_size) / sizeof (T);

	return item_off;
}

template < class T > 
TPIE_OS_OFFSET BTE_stream_mmap < T >::chunk_size () const {
	return header->block_size / sizeof (T);
}

#ifdef BTE_STREAM_MMAP_READ_AHEAD

template < class T > 
void BTE_stream_mmap < T >::read_ahead () {

	TPIE_OS_OFFSET f_curr_block;

	// The current block had better already be valid or we made a
	// mistake in being here.

	tp_assert (block_valid,
		"Trying to read ahead when current block is invalid.");

	// Check whether there is a next block.  If we are already in the
	// last block of the file then it makes no sense to read ahead.
	// What if we are writing?? Rajiv
	f_curr_block = ((f_offset - os_block_size_) / header->block_size) *
		header->block_size + os_block_size_;

	if (f_eos < f_curr_block + 2 * header->block_size) {
		return;			// XXX
		// need to fix this    
		// if not read only, we can extend the file and prefetch.
		// (only if not a substream)
		// Rajiv
		// prefetch only if write only and not substream
		if (!w_only || substream_level) {
#ifdef COLLECT_STATS
			stats_eos++;
#endif
			return;
		}
		if (w_only &&
			!substream_level &&
			(f_curr_block + 2 * header->block_size > f_filelen)) {
#ifdef VERBOSE
				if (verbose)
					cout << "growing file (fd" << fd << ") in advance\n";
#endif
				grow_file (f_curr_block);
			}
	}

	f_next_block = f_curr_block + header->block_size;

	// Rajiv
	assert (f_next_block + header->block_size <= f_filelen);
	assert (next_block != curr_block);
#if !USE_LIBAIO
	// took out the SYSTYPE_BSD ifdef for readability Rajiv
	next_block = (T *) (call_mmap (next_block, header->block_size,
		r_only, w_only,
		fd, f_next_block, (next_block != NULL)));
	assert (next_block != (T *) - 1);
	have_next_block = 1;
#endif				// !USE_LIBAIO

#if USE_LIBAIO
	// Asyncronously read the first word of each os block in the next
	// logical block.
	for (unsigned int ii = 0; ii < BTE_STREAM_MMAP_BLOCK_FACTOR; ii++) {

		// Make sure there is not a pending request for this block
		// before requesting it.

		if (aio_results[ii].aio_return != AIO_INPROGRESS) {
			aio_results[ii].aio_return = AIO_INPROGRESS;

			// We have to cancel the last one, even though it completed,
			// in order to allow another one with the same result.
			aiocancel (aio_results + ii);

			// Start the async I/O.
			if (aioread (fd, (char *) (read_ahead_buffer + ii), sizeof (int),
				f_next_block + ii * os_block_size_, SEEK_SET,
				aio_results + ii)) {

					os_errno = errno;

					TP_LOG_FATAL ("aioread() failed to read ahead");
					TP_LOG_FATAL ("\": ");
					TP_LOG_FATAL (strerror (os_errno));
					TP_LOG_FATAL ('\n');
					TP_LOG_FLUSH_LOG;
				}
		}
	}
#endif				// USE_LIBAIO
}

#endif				// BTE_STREAM_MMAP_READ_AHEAD

#undef BTE_STREAM_MMAP_MM_BUFFERS

#endif	// _BTE_STREAM_MMAP_H
