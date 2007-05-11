// Copyright (c) 1995 Darren Erik Vengroff
//
// File: ami_kb_dist.h
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 3/11/95
//
// $Id: ami_kb_dist.h,v 1.10 2004/08/12 12:35:30 jan Exp $
//
// Radix based distribution for single or striped AMI layers.
//

// Get definitions for working with Unix and Windows
#include <portability.h>

// If we have not already seen this file with KB_KEY undefined or
// KB_KEY is defined, we will process the file.

#if !(defined(_AMI_KB_DIST_H)) || defined(KB_KEY)

#ifdef KB_KEY

#define _KB_CONCAT(a,b) a ## b
#define _AMI_KB_DIST(kbk) _KB_CONCAT(AMI_kb_dist_,kbk)

#else

// KB_KEY is not defined, so set the flag so we won't come through
// this file again with KB_KEY unset, and set KB_KEY to kb_key
// temporarily.  We also Set the macro for the name of the function
// defined in this file.

#define _AMI_KB_DIST_H
#define KB_KEY kb_key

#ifdef _HAVE_TEMP_KB_KEY_DEFINITION_
#error _HAVE_TEMP_KB_KEY_DEFINITION_ already defined.
#else
#define _HAVE_TEMP_KB_KEY_DEFINITION_
#endif

#define _AMI_KB_DIST(kbk) AMI_kb_dist

#endif

#include <ami_stream.h>
#include <ami_key.h>

// This is a hack.  The reason it is here is that if AMI_STREAM<char>
// is used directly in the template for AMI_kb_dist() a parse error is
// generated at compile time.  I suspect this may be a bug in the
// template instantiation code in g++ 2.6.3.

#ifndef _DEFINED_TYPE_AMISC_
#define _DEFINED_TYPE_AMISC_
typedef AMI_STREAM<char> type_amisc;
#endif

template<class T>
AMI_err _AMI_KB_DIST(KB_KEY)(AMI_STREAM<T> &instream,
                             type_amisc &name_stream,
                             const key_range &range, TPIE_OS_OFFSET &max_size)
{
    AMI_err ae;

    size_t sz_avail;
    size_t single_stream_usage;

    // How many ouput streams will there be?
    unsigned int output_streams;
    
    unsigned int ii;
    
    // How much main memory do we have?
    sz_avail = MM_manager.memory_available ();

    // How much memory does a single stream need in the worst case?
    if ((ae = instream.main_memory_usage(&single_stream_usage,
                                         MM_STREAM_USAGE_MAXIMUM)) !=
                                         AMI_ERROR_NO_ERROR) {
        return ae;
    }

    // How many output streams can we buffer in that amount of space?
    // Recall that we also need a pointer and a range for each stream.
    output_streams = (unsigned int)((sz_avail - 2 * single_stream_usage) /
        (single_stream_usage + sizeof(AMI_STREAM<T> *) + sizeof(range)));
    
    // We need at least two output streams.
    if (output_streams < 2) {
        return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
    }

    // Make sure we don't use more streams than are available.
    {
        unsigned available_streams = instream.available_streams();

        if ((available_streams != (unsigned)-1) &&
            (available_streams < output_streams)) {
            output_streams = available_streams;
        }
    }

#ifdef AMI_RADIX_POWER_OF_TWO
    // Adjust the number of output streams so that it is a power of two.
    
#endif
    
    // Create the output streams and initialize the ranges they cover to
    // be empty.

    AMI_STREAM<T> **out_streams = new  AMI_STREAM<T> *[output_streams];
    key_range *out_ranges = new key_range[output_streams];
    
    for (ii = 0; ii < output_streams; ii++) {

        // This needs to be fixed to eliminate the max size parameter.
        // This should be done system-wide.
        out_streams[ii] = new AMI_STREAM<T>;

        out_ranges[ii].min = KEY_MAX;
        out_ranges[ii].max = KEY_MIN;
    }

    // Scan the input putting each item in the right output stream.

    instream.seek(0);

    unsigned int index_denom = (((range.max - range.min) / output_streams)
                                + 1);

    while (1) {
        T *in;
        kb_key k;
        
        ae = instream.read_item(&in);
        if (ae == AMI_ERROR_END_OF_STREAM) {
            break;
        } else if (ae != AMI_ERROR_NO_ERROR) {
            return ae;
        }

        k = (unsigned int)KB_KEY(*in);

#ifdef AMI_RADIX_POWER_OF_TWO
        // Do it with shifting and masking.
    
#else
        ii = (k - range.min) / index_denom;
#endif

        ae = out_streams[ii]->write_item(*in);
        if (ae != AMI_ERROR_NO_ERROR) {
            return ae;
        }

        if (k < out_ranges[ii].min) {
            out_ranges[ii].min = k;
        }
        if (k > out_ranges[ii].max) {
            out_ranges[ii].max = k;
        }
            
    }

    // Write the names and ranges of all non-empty output streams.

    for (ii = 0, max_size = 0; ii < output_streams; ii++) {
        char *stream_name;
        TPIE_OS_OFFSET stream_len;
        
        if ((stream_len = out_streams[ii]->stream_len()) > 0) {

            // cerr << stream_len << '\n';
            
            // Is it the biggest one so far?

            if (stream_len > max_size) {
                max_size = stream_len;
            }

            // Get the and write the name of the stream.

            ae = out_streams[ii]->name(&stream_name);
            if (ae != AMI_ERROR_NO_ERROR) {
                return ae;
            }

            ae = name_stream.write_array(stream_name,
                                         strlen(stream_name) + 1);
            if (ae != AMI_ERROR_NO_ERROR) {
                return ae;
            }

            // For purposes of efficiency, and to avoid having another
            // stream, we are going to cast the new range to an array of
            // characters and tack it onto the stream name stream.
            // We then do the same thing with the length of the stream.

            ae = name_stream.write_array((const char *)(out_ranges+ii),
                                         sizeof(key_range));
            if (ae != AMI_ERROR_NO_ERROR) {
                return ae;
            }
                
            ae = name_stream.write_array((const char *)&stream_len,
                                         sizeof(stream_len));
            if (ae != AMI_ERROR_NO_ERROR) {
                return ae;
            }
            
            // The semantics of name() are to allocate space for the
            // buffer it returns.  We are responsible for giving it back.
            delete [] stream_name;
    
            // Make this stream persist on disk since it is not empty.
            out_streams[ii]->persist(PERSIST_PERSISTENT);

        }

        // Delete the stream.
        delete out_streams[ii];

    }

    delete [] out_streams;
    
    // We're done.

    return AMI_ERROR_NO_ERROR;
}

#ifdef _HAVE_TEMP_KB_KEY_DEFINITION_
#undef _HAVE_TEMP_KB_KEY_DEFINITION_
#undef KB_KEY
#endif

#ifdef _KB_CONCAT
#undef _KB_CONCAT
#endif
#undef _AMI_KB_DIST

#endif // !(defined(_AMI_KB_DIST_H)) || defined(KB_KEY)
