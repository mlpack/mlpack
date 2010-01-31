//
// File: ami_stream_single.h (formerly ami_single.h)
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/19/94
//
// $Id: ami_stream_single.h,v 1.13 2005/07/07 20:43:06 adanner Exp $
//
// AMI entry points implemented on top of a single BTE.  This is useful
// for single CPU, single disk machines.
//
#ifndef _AMI_STREAM_SINGLE_H
#define _AMI_STREAM_SINGLE_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

// [tavi] for UINT_MAX
#include <limits.h>
#include <sys/types.h>

// Use tempnam() instead of mktemp().
// no - tempnam uses environment in way we dont like
#include <stdio.h>

// For free()
#include <stdlib.h>

// To make assertions.
#include "u/nvasil/tpie/tpie_assert.h"
#include <assert.h>

// Get an appropriate BTE.  Flags may have been set to determine
// exactly what BTE implementaion will be used, but that should be of
// little concern to us.  bte_stream.h and recursively included files
// will worry about parsing the appropriate flags and getting us an
// implementation.
#include "u/nvasil/tpie/bte_stream.h"

// Get the AMI_stream_base class.
#include "u/nvasil/tpie/ami_stream_base.h"

// Get the base memory manager. Normally the BTE will have already
// gotten this, but in library code where we have no BTE defined, it
// may not.
#include "u/nvasil/tpie/mm_base.h"

#include "u/nvasil/tpie/ami_device.h"
#include "u/nvasil/tpie/tpie_tempnam.h"

// An initializer class to set the default device for the
// AMI_stream_single_base class.
class AMI_stream_single_base_device_initializer {
private:
    static unsigned int count;
public:
    AMI_stream_single_base_device_initializer(void);
    ~AMI_stream_single_base_device_initializer(void);
};

// A base class for AMI single streams that is used to hold the
// default device description for AMI single streams regardless of the
// particular type of object in the stream.

class AMI_stream_single_base {
    friend AMI_stream_single_base_device_initializer::
        AMI_stream_single_base_device_initializer(void);
public:
    // The default device description for AMI streams.
    static AMI_device default_device;
    
    // The index into the device list for the next stream.    
    static unsigned int device_index;    

  static const tpie_stats_stream& gstats() 
    { return BTE_stream_base_generic::gstats(); }
};


// This is a trick to make sure that at least one initializer is declared.
// The constructor for this initializer will make sure that the default
// device is set up properly.
static AMI_stream_single_base_device_initializer one_ssbd_initializer_per_source_file;

// The single stream class.

template<class T> class AMI_stream_single : public AMI_stream_base<T>,
                                            public AMI_stream_single_base {
private:
  using AMI_stream_base<T>::status_;

  // Point to a base stream, since the particular type of BTE
  // stream we are using may vary.
  BTE_STREAM<T> *btes;
  
  int r_only;
  
  // Non-zero if we should destroy the bte stream when we the
  // AMI stream is destroyed.
  int destruct_bte;

public:
  
  // Read and write elements.
  A_INLINE AMI_err read_item(T **elt);
  A_INLINE AMI_err write_item(const T &elt);
  
  A_INLINE AMI_err read_array(T *mm_space, TPIE_OS_OFFSET *len);
  A_INLINE AMI_err write_array(const T *mm_space, TPIE_OS_OFFSET len);
  
  // We have a variety of constructors for different uses.
  
  // A temporary AMI_stream using the default BTE stream type and
  // a temporary space on the disk or in some file system.
  AMI_stream_single(unsigned int device = UINT_MAX);
  
  // An AMI stream based on a specific path name.
  AMI_stream_single(const char *path_name, 
		    AMI_stream_type st = AMI_READ_WRITE_STREAM);
  
  // An AMI stream based on a specific existing BTE stream.  Note
  // that in this case the BTE stream will not be detroyed when the
  // destructor is called.
  AMI_stream_single(BTE_STREAM<T> *bs);
  
  // A psuedo-constructor for substreams.
  AMI_err new_substream(AMI_stream_type st, TPIE_OS_OFFSET sub_begin, 
			TPIE_OS_OFFSET sub_end,
			AMI_stream_base<T> **sub_stream);
  
  // Return the number of items in the stream.
  TPIE_OS_OFFSET stream_len(void) const { return btes->stream_len(); }
  
  // Return the path name of this stream in newly allocated space.
  AMI_err name(char **stream_name);
  
  // Move to a specific position in the stream.
  AMI_err seek(TPIE_OS_OFFSET offset);
  
  // Return the current position in the stream.
  TPIE_OS_OFFSET tell() const { return btes->tell(); }

  // Truncate
  AMI_err truncate(TPIE_OS_OFFSET offset);
  
  // Query memory usage
  AMI_err main_memory_usage(TPIE_OS_SIZE_T *usage,
			    MM_stream_usage usage_type);
  
  // Destructor
  ~AMI_stream_single(void);
  
  const tpie_stats_stream& stats() const { return btes->stats(); }

  int available_streams(void);
  
  TPIE_OS_OFFSET chunk_size(void) const { return btes->chunk_size(); }
  
  void persist(persistence p);
  
  persistence persist() const { return btes->persist(); }

  char *sprint();
};


// Create a temporary AMI stream on one of the devices in the default
// device description. Persistence is PERSIST_DELETE by default. We
// are given the index of the string describing the desired device.
template<class T>
AMI_stream_single<T>::AMI_stream_single(unsigned int device) {

  // [tavi] Hack to fix an error that appears in gcc 2.8.1
  if (device == UINT_MAX) {
    device = (device_index = ((device_index + 1) % default_device.arity()));
  }
  
  r_only = 0;
  destruct_bte = 1;
  
  // Get a unique name.
  char *path = tpie_tempnam("AMI", default_device[device]);
  
 TP_LOG_DEBUG_ID("Temporary stream in file: ");
 TP_LOG_DEBUG_ID(path);
  
  // Create the BTE stream.
  btes = new BTE_STREAM<T>(path, BTE_WRITE_STREAM);
  
  // (Short circuit evaluation...)
  if (btes == NULL || btes->status() == BTE_STREAM_STATUS_INVALID) {
   TP_LOG_FATAL_ID("BTE returned invalid or NULL stream.");
    status_ = AMI_STREAM_STATUS_INVALID;
    return;
  }
  
  btes->persist(PERSIST_DELETE);
  
  if (seek(0) != AMI_ERROR_NO_ERROR) {
   TP_LOG_FATAL_ID("seek(0) returned error.");
    status_ = AMI_STREAM_STATUS_INVALID;
    return;
  }
  
  status_ = AMI_STREAM_STATUS_VALID;
};


// A stream created with this constructor will persist on disk at the
// location specified by the path name.
template<class T>
AMI_stream_single<T>::AMI_stream_single(const char *path_name,
					AMI_stream_type st) {

  // Decide BTE stream type
  BTE_stream_type bst;
  switch (st) {
  case AMI_READ_STREAM: 
    bst = BTE_READ_STREAM;
    break;
  case AMI_APPEND_STREAM:
    bst = BTE_APPEND_STREAM;
    break;
  case AMI_WRITE_STREAM:
  case AMI_READ_WRITE_STREAM:
    bst = BTE_WRITE_STREAM; //BTE_WRITE_STREAM means both read and
 //write; this is inconsistent and should be modified..
    break;
  default:
   TP_LOG_WARNING_ID("Unknown stream type passed to constructor;");
   TP_LOG_WARNING_ID("Defaulting to AMI_READ_WRITE_STREAM.");
    bst = BTE_WRITE_STREAM;
    break;
  }
  
  r_only = ((st == AMI_READ_STREAM)? 1 : 0);
  destruct_bte = 1;
  
  // Create the BTE stream.
  btes = new BTE_STREAM<T>(path_name, bst);
  // (Short circuit evaluation...)
  if (btes == NULL || btes->status() == BTE_STREAM_STATUS_INVALID) {
   TP_LOG_FATAL_ID("BTE returned invalid or NULL stream.");
    status_ = AMI_STREAM_STATUS_INVALID;
    return;
  }

  btes->persist(PERSIST_PERSISTENT);

  // If an APPEND stream, the BTE constructor seeks to its end;
  if (st != AMI_APPEND_STREAM) {
    if (seek(0) != AMI_ERROR_NO_ERROR) {
     TP_LOG_FATAL_ID("seek(0) returned error.");
      status_ = AMI_STREAM_STATUS_INVALID;
      return;
    }
  }
  
  status_ = AMI_STREAM_STATUS_VALID;
};


template<class T>
AMI_stream_single<T>::AMI_stream_single(BTE_STREAM<T> *bs) {

  destruct_bte = 0;

  btes = bs;
  if (btes == NULL || btes->status() == BTE_STREAM_STATUS_INVALID) {
   TP_LOG_FATAL_ID("BTE returned invalid or NULL stream.");
    status_ = AMI_STREAM_STATUS_INVALID;
    return;
  }

  r_only = bs->read_only();
  status_ = AMI_STREAM_STATUS_VALID;
};


template<class T>
AMI_err AMI_stream_single<T>::new_substream(AMI_stream_type st,
                                            TPIE_OS_OFFSET sub_begin,
                                            TPIE_OS_OFFSET sub_end,
                                            AMI_stream_base<T> **sub_stream)
{
    AMI_err ae = AMI_ERROR_NO_ERROR;
    // Check permissions. Only READ and WRITE are allowed, and only READ is
    // allowed if r_only is set.
    if ((st != AMI_READ_STREAM) && ((st != AMI_WRITE_STREAM) || r_only)) {
        *sub_stream = NULL;
		TP_LOG_DEBUG_ID("permission denied");		
        return AMI_ERROR_PERMISSION_DENIED;
    }
    
    BTE_stream_base<T> *bte_ss;

    if (btes->new_substream(((st == AMI_READ_STREAM) ? BTE_READ_STREAM :
                             BTE_WRITE_STREAM),
                             sub_begin, sub_end,
                             &bte_ss) != BTE_ERROR_NO_ERROR) {
	 TP_LOG_DEBUG_ID("new_substream failed");		
	  *sub_stream = NULL;
	  return AMI_ERROR_BTE_ERROR;
    }

    AMI_stream_single<T> *ami_ss;

    // This is a potentially dangerous downcast.  It is being done for
    // the sake of efficiency, so that calls to the BTE can be
    // inlined.  If multiple implementations of BTE streams are
    // present it could be very dangerous.

    // this is to avoid compiler warnings
	// hack! XXX
#if(0)
	BTE_STREAM<T> *bte_ss_b=0;
	assert(sizeof(BTE_STREAM<T>*) == sizeof(BTE_stream_base<T>*));
	memcpy(bte_ss_b, bte_ss, sizeof(BTE_STREAM<T>*));
    ami_ss = new AMI_stream_single<T>(bte_ss_b);
#endif
    ami_ss = new AMI_stream_single<T>((BTE_STREAM<T>*)bte_ss);

    ami_ss->destruct_bte = 1;
	ae = ami_ss->seek(0);
	assert(ae == AMI_ERROR_NO_ERROR); // sanity check

    *sub_stream = (AMI_stream_base<T> *)ami_ss;

    return ae;
}


template<class T>
AMI_err AMI_stream_single<T>::name(char **stream_name)
{
    BTE_err be = btes->name(stream_name);
    if (be != BTE_ERROR_NO_ERROR) {
	 TP_LOG_WARNING_ID("bte error");
	  return AMI_ERROR_BTE_ERROR;
    } else {
	  return AMI_ERROR_NO_ERROR;
    }
}

// Move to a specific offset.
template<class T>
AMI_err AMI_stream_single<T>::seek(TPIE_OS_OFFSET offset)
{
    if (btes->seek(offset) != BTE_ERROR_NO_ERROR) {
	 TP_LOG_WARNING_ID("bte error");		
	  return AMI_ERROR_BTE_ERROR;
    }

    return AMI_ERROR_NO_ERROR;
}

// Truncate
template<class T>
AMI_err AMI_stream_single<T>::truncate(TPIE_OS_OFFSET offset)
{
    if (btes->truncate(offset) != BTE_ERROR_NO_ERROR) {
	 TP_LOG_WARNING_ID("bte error");
	  return AMI_ERROR_BTE_ERROR;
    }

    return AMI_ERROR_NO_ERROR;
}

// Query memory usage
template<class T>
AMI_err AMI_stream_single<T>::main_memory_usage(TPIE_OS_SIZE_T *usage,
                                                MM_stream_usage usage_type)
{
    if (btes->main_memory_usage(usage, usage_type) != BTE_ERROR_NO_ERROR) {
	 TP_LOG_WARNING_ID("bte error");		
	  return AMI_ERROR_BTE_ERROR;
    }

    switch (usage_type) {
        case MM_STREAM_USAGE_OVERHEAD:
        case MM_STREAM_USAGE_CURRENT:
        case MM_STREAM_USAGE_MAXIMUM:
        case MM_STREAM_USAGE_SUBSTREAM:
            *usage += sizeof(*this);
            break;
        case MM_STREAM_USAGE_BUFFER:
            break;
        default:
            tp_assert(0, "Unknown MM_stream_usage type added.");
            break;
    }

    return AMI_ERROR_NO_ERROR;
}

template<class T>
AMI_stream_single<T>::~AMI_stream_single(void)
{
    if (destruct_bte) {
        delete btes;
    }
}

template<class T>
A_INLINE AMI_err AMI_stream_single<T>::read_item(T **elt) 
{
    BTE_err bte_err;
	AMI_err ae;

	bte_err = btes->read_item(elt);
	switch(bte_err) {
	case BTE_ERROR_NO_ERROR:
	  ae = AMI_ERROR_NO_ERROR;
	  break;
	case BTE_ERROR_END_OF_STREAM:
	 TP_LOG_DEBUG_ID("eos in read_item");
	  ae = AMI_ERROR_END_OF_STREAM;
	  break;
	default:
	 TP_LOG_DEBUG_ID("bte error in read_item");
	  ae = AMI_ERROR_BTE_ERROR;
	  break;
	}
	return ae;
}

template<class T>
A_INLINE AMI_err AMI_stream_single<T>::write_item(const T &elt)
{
    if (btes->write_item(elt) != BTE_ERROR_NO_ERROR) {
	 TP_LOG_WARNING_ID("bte error");
	  return AMI_ERROR_BTE_ERROR;
    }
	return AMI_ERROR_NO_ERROR;
}


template<class T>
A_INLINE AMI_err AMI_stream_single<T>::read_array(T *mm_space, TPIE_OS_OFFSET *len)
{
    BTE_err be;
    T *read;
    TPIE_OS_OFFSET ii;
    
    // How long is it.
    TPIE_OS_OFFSET str_len = *len;
            
    // Read them all.
    for (ii = str_len; ii--; ) {
        if ((be = btes->read_item(&read)) != BTE_ERROR_NO_ERROR) {
            if (be == BTE_ERROR_END_OF_STREAM) {
                return AMI_ERROR_END_OF_STREAM;
            } else { 
                return AMI_ERROR_BTE_ERROR;
            }
        }
        *mm_space++ = *read;
    }

    *len = str_len;
    
    return AMI_ERROR_NO_ERROR;
}

template<class T>
A_INLINE AMI_err AMI_stream_single<T>::write_array(const T *mm_space, TPIE_OS_OFFSET len)
{
    BTE_err be;
    TPIE_OS_OFFSET ii;
    
    for (ii = len; ii--; ) {
        if ((be = btes->write_item(*mm_space++)) != BTE_ERROR_NO_ERROR) {
            if (be == BTE_ERROR_END_OF_STREAM) {
                return AMI_ERROR_END_OF_STREAM;
            } else { 
                return AMI_ERROR_BTE_ERROR;
            }
        }
    }
    return AMI_ERROR_NO_ERROR;
}        

template<class T>
int AMI_stream_single<T>::available_streams(void)
{
    return btes->available_streams();
}


template<class T>
void AMI_stream_single<T>::persist(persistence p)
{
    btes->persist(p);
}

// sprint()
// Return a string describing the stream
//
// This function gives easy access to the file name, length.
// It is not reentrant, but this should not be too much of a problem 
// if you are careful.
template<class T>
char *AMI_stream_single<T>::sprint()
{
  static char buf[BUFSIZ];
  char *s;
  name(&s);
  sprintf(buf, "[AMI_STREAM %s %ld]", s, (long)stream_len());
  delete s;
  return buf;
}

#endif // _AMI_STREAM_SINGLE_H 
