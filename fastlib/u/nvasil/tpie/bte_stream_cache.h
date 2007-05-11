//
// File: bte_stream_cache.h (formerly bte_cache.h)
// Author: Darren Erik Vengroff <darrenv@eecs.umich.edu>
// Created: 9/19/94
//
// $Id: bte_stream_cache.h,v 1.4 2004/08/12 12:35:31 jan Exp $
//
// BTE streams for main memory caches.
//
#ifndef _BTE_STREAM_CACHE_H
#define _BTE_STREAM_CACHE_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// Include the registration based memory manager.
#define MM_IMP_REGISTER
#include <mm.h>

#include <bte_stream_base.h>

// This code makes assertions and logs errors.
#include <tpie_assert.h>
#include <tpie_log.h>


#define BTE_STREAM_CACHE_DEFAULT_MAX_LEN (1024 * 256)

#ifndef BTE_STREAM_CACHE_LINE_SIZE
#define BTE_STREAM_CACHE_LINE_SIZE 64
#endif // BTE_STREAM_CACHE_LINE_SIZE

//
// The cache stream class.
//

template <class T>
class BTE_stream_cache : public BTE_stream_base<T> {
private:
    T *data;
    T *current;
    T *data_max;
    T *data_hard_end;
    unsigned int substream_level;
    unsigned int valid;
    unsigned int r_only;
    
    BTE_stream_cache(void);
public:

    // Constructors
    BTE_stream_cache(const char *path, BTE_stream_type st, TPIE_OS_OFFSET max_len); 

    // A psuedo-constructor for substreams.
    BTE_err new_substream(BTE_stream_type st, TPIE_OS_OFFSET sub_begin,
                          TPIE_OS_OFFSET sub_end, BTE_stream_base<T> **sub_stream);
    

    // Query memory usage
    BTE_err main_memory_usage(size_t *usage,
                              MM_stream_usage usage_type);

    // Return the number of items in the stream.
    TPIE_OS_OFFSET stream_len(void);

    // Move to a specific position in the stream.
    BTE_err seek(TPIE_OS_OFFSET offset);
    
    // Destructor
    ~BTE_stream_cache(void);

    BTE_err read_item(T **elt);
    BTE_err write_item(const T &elt);
    int read_only(void) { return r_only; };

    int available_streams(void) { return -1; };    

    TPIE_OS_OFFSET chunk_size(void);
};


template<class T>
BTE_stream_cache<T>::BTE_stream_cache(void)
{
};

template<class T>
BTE_stream_cache<T>::BTE_stream_cache(const char *path, BTE_stream_type st,
                                      TPIE_OS_OFFSET max_len) {

    // A stream being created out of the blue must be writable, so we
    // return an error if it is not.

    switch (st) {
        case BTE_READ_STREAM:
        case BTE_APPEND_STREAM:
            valid = 0;
            break;
        case BTE_WRITE_STREAM:
            r_only = 0;
            if (!max_len) {
                max_len = BTE_STREAM_CACHE_DEFAULT_MAX_LEN;
            }
            // Use malloc() directly rather than new becasue this is
            // in "secondary memory" and will not necessarily go into
            // the cache.
            data = (T*)malloc(max_len*sizeof(T));
            if (data == NULL) {
                valid = 0;
               TP_LOG_FATAL_ID("Out of \"secondary memory.\"");
                return;
            }
            current = data_max = data;
            data_hard_end = data + max_len;
            valid = 1;
    }
};


template<class T>
BTE_err BTE_stream_cache<T>::new_substream(BTE_stream_type st, TPIE_OS_OFFSET sub_begin,
                                           TPIE_OS_OFFSET sub_end,
                                           BTE_stream_base<T> **sub_stream)
{
    BTE_stream_cache *ss;
    
    if (st == BTE_APPEND_STREAM) {
            return BTE_ERROR_PERMISSION_DENIED;
    } else {
        if ((sub_begin >= data_hard_end - data) ||
            (sub_end >= data_hard_end - data) ||
            (sub_begin >= data_max - data) ||
            (sub_end >= data_max - data) ||
            (sub_end < sub_begin)) {
            return BTE_ERROR_OFFSET_OUT_OF_RANGE;
        }
        ss = new BTE_stream_cache;
        ss->r_only = (st == BTE_READ_STREAM);
        ss->substream_level = substream_level + 1;
        ss->current = ss->data = data + sub_begin;
        ss->data_max = ss->data_hard_end = data + sub_end + 1;
        *sub_stream = (BTE_stream_base<T> *)ss;                       
        return BTE_ERROR_NO_ERROR;
    }
};           
            

template<class T>
BTE_err BTE_stream_cache<T>::main_memory_usage(size_t *usage,
                                               MM_stream_usage usage_type)
{
    switch (usage_type) {
        case MM_STREAM_USAGE_CURRENT:
        case MM_STREAM_USAGE_MAXIMUM:
        case MM_STREAM_USAGE_SUBSTREAM:
            *usage = sizeof(*this) + BTE_STREAM_CACHE_LINE_SIZE;
            break;
        case MM_STREAM_USAGE_BUFFER:
            *usage = BTE_STREAM_CACHE_LINE_SIZE;
            break;
        case MM_STREAM_USAGE_OVERHEAD:
            *usage = sizeof(this);
            break;
    }
    return BTE_ERROR_NO_ERROR;
};


template<class T>
TPIE_OS_OFFSET BTE_stream_cache<T>::stream_len(void)
{
    return data_max - data;
};



template<class T>
BTE_err BTE_stream_cache<T>::seek(TPIE_OS_OFFSET offset)
{
    if (offset > data_hard_end - data) {
        return BTE_ERROR_OFFSET_OUT_OF_RANGE;
    } else {
        current = data + offset;
        return BTE_ERROR_NO_ERROR;
    }
};


template<class T>
BTE_stream_cache<T>::~BTE_stream_cache(void)
{
    if (!substream_level) {
        delete data;
    }
};

template<class T>
BTE_err BTE_stream_cache<T>::read_item(T **elt)
{
    if (current >= data_max) {
        return BTE_ERROR_END_OF_STREAM;
    } else {
        *elt = current++;
        return BTE_ERROR_NO_ERROR;
    }
};

template<class T>
BTE_err BTE_stream_cache<T>::write_item(const T &elt)
{
    if (r_only) {
        return BTE_ERROR_PERMISSION_DENIED;
    }
    if (current >= data_hard_end) {
        return BTE_ERROR_OUT_OF_SPACE;
    } else {
        *current++ = elt;
        if (current > data_max) {
            data_max = current;
        }
        return BTE_ERROR_NO_ERROR;
    }   
};


template<class T>
TPIE_OS_OFFSET BTE_stream_cache<T>::chunk_size(void)
{
    return BTE_STREAM_CACHE_LINE_SIZE / sizeof(T);
}


#endif // _BTE_STREAM_CACHE_H 
