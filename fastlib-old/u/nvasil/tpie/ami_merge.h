//
// File: ami_merge.h
// Author: Darren Erik Vengroff <dev@cs.duke.edu>
// Created: 5/31/94
//
// First cut at a merger.  Obviously missing is code to verify that
// lower level streams will use appropriate levels of buffering.  This
// will be more critical for parallel disk implementations.
//
// $Id: ami_merge.h,v 1.38 2005/07/07 20:43:36 adanner Exp $
//
#ifndef _AMI_MERGE_H
#define _AMI_MERGE_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"

// For log() and such as needed to compute tree heights.
#include <math.h>

#include "u/nvasil/tpie/ami_stream.h"

enum AMI_merge_output_type {
    AMI_MERGE_OUTPUT_OVERWRITE = 1,
    AMI_MERGE_OUTPUT_APPEND
};

typedef int AMI_merge_flag;
typedef int arity_t;

#define CONST const


// CLASSES AND FUNCTIONS DEFINED IN THIS MODULE
//------------------------------------------------------------

//A superclass for merge management objects
template<class T> class AMI_generalized_merge_base;


//merge <arity> streams using a merge management object and write
//result into <outstream>; it is assumed that the available memory can
//fit the <arity> streams, the output stream and also the space
//required by the merge management object; AMI_generalized_merge() checks this and
//then calls AMI_generalized_single_merge();
template<class T, class M>
AMI_err AMI_generalized_merge(AMI_STREAM<T> **instreams, arity_t arity,
		  AMI_STREAM<T> *outstream, M *m_obj);


// divide the input stream in substreams, merge each substream
// recursively, and merge them together using AMI_generalized_single_merge()
template<class T, class M>
AMI_err AMI_generalized_partition_and_merge(AMI_STREAM<T> *instream,
                                AMI_STREAM<T> *outstream, M *m_obj);


//merge <arity> streams in memory using a merge management object and
//write result into <outstream>; 
template<class T, class M>
AMI_err  AMI_generalized_single_merge(AMI_STREAM<T> **instreams, arity_t arity,
			  AMI_STREAM<T> *outstream, M *m_obj);


//read <instream> in memory and merge it using
//m_obj->main_mem_operate(); if <instream> does not fit in main memory
//return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
template<class T, class M>
AMI_err AMI_main_mem_merge(AMI_STREAM<T> *instream,
                           AMI_STREAM<T> *outstream, M *m_obj);

//------------------------------------------------------------







//------------------------------------------------------------
// A superclass for merge management objects
//------------------------------------------------------------
template<class T>
class AMI_generalized_merge_base {
public:

#if AMI_VIRTUAL_BASE
  virtual AMI_err initialize(arity_t arity,
			     CONST T * CONST * in,
			     AMI_merge_flag *taken_flags,
			     int &taken_index) = 0;
  virtual AMI_err operate(CONST T * CONST *in,
			  AMI_merge_flag *taken_flags,
			  int &taken_index,
			  T *out) = 0;
  virtual AMI_err main_mem_operate(T* mm_stream, TPIE_OS_SIZE_T len) = 0;
  virtual TPIE_OS_SIZE_T space_usage_overhead(void) = 0;
  virtual TPIE_OS_SIZE_T space_usage_per_stream(void) = 0;
#endif // AMI_VIRTUAL_BASE

};





//------------------------------------------------------------ 

//merge <arity> streams using a merge management object and write
//result into <outstream>; it is assumed that the available memory can
//fit the <arity> streams, the output stream and also the space
//required by the merge management object; AMI_generalized_merge() checks this and
//then calls AMI_generalized_single_merge();

//------------------------------------------------------------ 
template<class T, class M>
AMI_err 
AMI_generalized_merge(AMI_STREAM<T> **instreams, arity_t arity,
	  AMI_STREAM<T> *outstream, M *m_obj) {

  TPIE_OS_SIZE_T sz_avail;
  TPIE_OS_OFFSET sz_stream, sz_needed = 0;
  
  // How much main memory is available?
  sz_avail = MM_manager.memory_available ();

  // Iterate through the streams, finding out how much additional
  // memory each stream will need in the worst case (the streams are
  // in memory, but their memory usage could be smaller then the
  // maximum one; one scenario is when the streams have been loaded
  // from disk with no subsequent read_item/write_item operation, in
  // which case their current memory usage is just the header block);
  // count also the output stream
  for (unsigned int ii = 0; ii < arity + 1; ii++) {
    instreams[ii]->main_memory_usage(&sz_stream, MM_STREAM_USAGE_MAXIMUM);
    sz_needed += sz_stream;
    instreams[ii]->main_memory_usage(&sz_stream, MM_STREAM_USAGE_CURRENT);
    sz_needed -= sz_stream;
  }                              
  
  //count the space used by the merge_management object (include
  //overhead added to a stream)
  sz_needed += m_obj->space_usage_overhead() + 
               arity * m_obj->space_usage_per_stream();
               
  //streams and m_obj must fit in memory!
  if (sz_needed >= (TPIE_OS_OFFSET)sz_avail) {
   TP_LOG_WARNING("Insuficent main memory to perform a merge.\n");
    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
  }
  assert(sz_needed < sz_avail);
  
  //merge streams in memory
  return AMI_generalized_single_merge(instreams, arity, outstream, m_obj);
};





//------------------------------------------------------------

//merge <arity> streams in memory using a merge management object and
//write result into <outstream>; 

//------------------------------------------------------------
template<class T, class M>
AMI_err 
AMI_generalized_single_merge(AMI_STREAM<T> **instreams, arity_t arity,
		 AMI_STREAM<T> *outstream, M *m_obj) {

  unsigned int ii;
  AMI_err ami_err;
  
  // Create an array of pointers for the input.
  T* *in_objects = new T*[arity];
  
  // Create an array of flags the merge object can use to ask for more
  // input from specific streams.
  AMI_merge_flag* taken_flags  = new AMI_merge_flag[arity];
  
  // An index to speed things up when the merge object takes only from
  // one index.
  int taken_index;
  
  //Output of the merge object.
  T merge_out;
  
#if DEBUG_PERFECT_MERGE
  unsigned int input_count = 0, output_count = 0;
#endif    
  
  // Rewind and read the first item from every stream; count the
  // number of non-null items read
  for (ii = arity; ii--; ) {
    if ((ami_err = instreams[ii]->seek(0)) != AMI_ERROR_NO_ERROR) {
	delete[] in_objects;
	delete[] taken_flags;
	return ami_err;
    }
    if ((ami_err = instreams[ii]->read_item(&(in_objects[ii]))) !=
	AMI_ERROR_NO_ERROR) {
      //error on read
   if (ami_err == AMI_ERROR_END_OF_STREAM) {
	in_objects[ii] = NULL;
      } else {
	delete[] in_objects;
	delete[] taken_flags;
	return ami_err;
      }
      // Set the taken flags to 0 before we call intialize()
      taken_flags[ii] = 0;
    } else {
      //item read succesfully
#if DEBUG_PERFECT_MERGE
      input_count++;
#endif                
    }
  }
  
  // Initialize the merge object.
  if (((ami_err = m_obj->initialize(arity, in_objects, taken_flags, 
				    taken_index)) != AMI_ERROR_NO_ERROR) &&
      (ami_err != AMI_MERGE_READ_MULTIPLE)) {
    return AMI_ERROR_OBJECT_INITIALIZATION;
  }      
  
  
  // Now simply call the merge object repeatedly until it claims to
  // be done or generates an error.
  while (1) {
    if (ami_err == AMI_MERGE_READ_MULTIPLE) {
      for (ii = arity; ii--; ) {
	if (taken_flags[ii]) {
	  ami_err = instreams[ii]->read_item(&(in_objects[ii]));
	  if (ami_err != AMI_ERROR_NO_ERROR) {
	    if (ami_err == AMI_ERROR_END_OF_STREAM) {
	      in_objects[ii] = NULL;
	    } else {
		delete[] in_objects;
		delete[] taken_flags;
		return ami_err;
	    }
	  } else {
#if DEBUG_PERFECT_MERGE                    
	    input_count++;
#endif
	  }
	}
	// Clear all flags before operate is called.
	taken_flags[ii] = 0;
      }
    } else {
      // The last call took at most one item.
      if (taken_index >= 0) {
	ami_err = instreams[taken_index]->
	  read_item(&(in_objects[taken_index]));
	if (ami_err != AMI_ERROR_NO_ERROR) {
	  if (ami_err == AMI_ERROR_END_OF_STREAM) {
	    in_objects[taken_index] = NULL;
	  } else {
	      delete[] in_objects;
	      delete[] taken_flags;
	      return ami_err;
	  }
	} else {
#if DEBUG_PERFECT_MERGE                    
	  input_count++;
#endif
	}
	taken_flags[taken_index] = 0;
      }
    }
    ami_err = m_obj->operate(in_objects, taken_flags, taken_index,
			     &merge_out);
    if (ami_err == AMI_MERGE_DONE) {
      break;
    } else if (ami_err == AMI_MERGE_OUTPUT) {
#if DEBUG_PERFECT_MERGE
      output_count++;
#endif                    
      if ((ami_err = outstream->write_item(merge_out)) !=
	  AMI_ERROR_NO_ERROR) {
	  delete[] in_objects;
	  delete[] taken_flags;
	  return ami_err;
      }            
    } else if ((ami_err != AMI_MERGE_CONTINUE) &&
	       (ami_err != AMI_MERGE_READ_MULTIPLE)) {
	delete[] in_objects;
	delete[] taken_flags;
	return ami_err;
    }
  }
  
#if DEBUG_PERFECT_MERGE
  tp_assert(input_count == output_count,
	    "Merge done, input_count = " << input_count <<
	    ", output_count = " << output_count << '.');
#endif

  delete[] in_objects;
  delete[] taken_flags;

  return AMI_ERROR_NO_ERROR;
};





//------------------------------------------------------------

//read <instream> in memory and merge it using
//m_obj->main_mem_operate(); if <instream> does not fit in main memory
//return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;

//------------------------------------------------------------
template<class T, class M>
AMI_err AMI_main_mem_merge(AMI_STREAM<T> *instream,
                           AMI_STREAM<T> *outstream, M *m_obj)  {

  AMI_err ae;
  TPIE_OS_OFFSET len;
  TPIE_OS_SIZE_T sz_avail;
  
  // How much memory is available?
  sz_avail = MM_manager.memory_available ();

  len = instream->stream_len();
  if ((len * sizeof(T)) <= (TPIE_OS_OFFSET)sz_avail) {
    
    // If the whole input can fit in main memory just call
    // m_obj->main_mem_operate
    
    ae = instream->seek(0);
    assert(ae == AMI_ERROR_NO_ERROR);
    
    // This code is sloppy and has to be rewritten correctly for
    // parallel buffer allocation.  It will not work with anything
    // other than a registration based memory manager.
    T *mm_stream;
    TPIE_OS_OFFSET len1;
    //allocate and read input stream in memory we know it fits, so we may cast.
    if ((mm_stream = new T[(TPIE_OS_SIZE_T)len]) == NULL) {
      return AMI_ERROR_MM_ERROR;
    };
    len1 = len;
    if ((ae = instream->read_array(mm_stream, &len1)) !=
	AMI_ERROR_NO_ERROR) {
      return ae;
    }
    tp_assert(len1 == len, "Did not read the right amount; "
	      "Allocated space for " << len << ", read " << len1 << '.');
    
    //just call m_obj->main_mem_operate. We know that len items fit into
	//main memory, so we may cast to TPIE_OS_SIZE_T
    if ((ae = m_obj->main_mem_operate(mm_stream, (TPIE_OS_SIZE_T)len)) !=
	AMI_ERROR_NO_ERROR) {
     TP_LOG_WARNING_ID("main_mem_operate failed");
      return ae;
    }

    //write array back to stream
    if ((ae = outstream->write_array(mm_stream, (TPIE_OS_SIZE_T)len)) !=
	AMI_ERROR_NO_ERROR) {
     TP_LOG_WARNING_ID("write array failed");
      return ae;
    }

    delete [] mm_stream;
    return AMI_ERROR_NO_ERROR;
    
  } else {
    
    // Something went wrong.  We should not have called this
    // function, since we don't have enough main memory.
   TP_LOG_WARNING_ID("out of memory");
    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
  }
};






//------------------------------------------------------------

// divide the input stream in substreams, merge each substream
// recursively, and merge them together using AMI_generalized_single_merge()

//------------------------------------------------------------
template<class T, class M>
AMI_err AMI_generalized_partition_and_merge(AMI_STREAM<T> *instream,
                                AMI_STREAM<T> *outstream, M *m_obj) {

  AMI_err ae;
  TPIE_OS_OFFSET len;
  TPIE_OS_SIZE_T sz_avail, sz_stream;
  unsigned int ii;
  int jj;
  
  //How much memory is available?
  sz_avail = MM_manager.memory_available ();

  // If the whole input can fit in main memory then just call
  // AMI_main_mem_merge() to deal with it by loading it once and
  // processing it.
  len = instream->stream_len();
  if ((len * sizeof(T)) <= (TPIE_OS_OFFSET)sz_avail) {
    return AMI_main_mem_merge(instream, outstream, m_obj);
  } 
  //else {

  
 
  // The number of substreams that can be merged together at once; i
  // this many substreams (at most) we are dividing the input stream
  arity_t merge_arity;
  
  //nb of substreams the original input stream will be split into
  arity_t nb_orig_substr;
  
  // length (nb obj of type T) of the original substreams of the input
  // stream.  The last one may be shorter than this.
  TPIE_OS_OFFSET sz_orig_substr;
  
  // The initial temporary stream, to which substreams of the
  // original input stream are written.
  AMI_STREAM<T> *initial_tmp_stream;

  // A pointer to the buffer in main memory to read a memory load into.
  T *mm_stream;
  
  
  // Loop variables:
  
  // The stream being read at the current level.
  AMI_STREAM<T> *current_input;

  // The output stream for the current level if it is not outstream.
  AMI_STREAM<T> *intermediate_tmp_stream;
        
  // The size of substreams of *current_input that are being
  // merged.  The last one may be smaller.  This value should be
  // sz_orig_substr * (merge_arity ** k) where k is the
  // number of iterations the loop has gone through.
  TPIE_OS_OFFSET current_substream_len;

  // The exponenent used to verify that current_substream_len is
  // correct.
  unsigned int k;
  
  TPIE_OS_OFFSET sub_start, sub_end;
  
  
  
  // How many substreams will there be?  The main memory
  // available to us is the total amount available, minus what
  // is needed for the input stream and the temporary stream.
  if ((ae = instream->main_memory_usage(&sz_stream, MM_STREAM_USAGE_MAXIMUM)) 
      != AMI_ERROR_NO_ERROR) {
    return ae;
  }                                     
  if (sz_avail <= 2 * sz_stream + sizeof(T)) {
    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
  }
  sz_avail -= 2 * sz_stream;

	
  // number of elements that will fit in memory (M) -R
  sz_orig_substr = sz_avail / sizeof(T);

  // Round the original substream length off to an integral number of
  // chunks.  This is for systems like HP-UX that cannot map in
  // overlapping regions.  It is also required for BTE's that are
  // capable of freeing chunks as they are read.
  {
    TPIE_OS_OFFSET sz_chunk_size = instream->chunk_size();
    
    sz_orig_substr = sz_chunk_size *
      ((sz_orig_substr + sz_chunk_size - 1) /sz_chunk_size);
    // WARNING sz_orig_substr now may not fit in memory!!! -R
  }

  // number of memoryloads in input ceil(N/M) -R
  nb_orig_substr = (arity_t)((len + sz_orig_substr - 1) / sz_orig_substr);
  
  // Account for the space that a merge object will use.
  {
    TPIE_OS_SIZE_T sz_avail_during_merge = sz_avail - m_obj->space_usage_overhead();
    TPIE_OS_SIZE_T sz_stream_during_merge = sz_stream +m_obj->space_usage_per_stream();
    
    merge_arity = (arity_t)((sz_avail_during_merge +
		   sz_stream_during_merge - 1) / sz_stream_during_merge);
  }

  // Make sure that the AMI is willing to provide us with the number
  // of substreams we want.  It may not be able to due to operating
  // system restrictions, such as on the number of regions that can be
  // mmap()ed in.
  {
    int ami_available_streams = instream->available_streams();
    
    if (ami_available_streams != -1) {
      if (ami_available_streams <= 4) {
	return AMI_ERROR_INSUFFICIENT_AVAILABLE_STREAMS;
      }
      if (merge_arity > (arity_t)ami_available_streams - 2) {
	merge_arity = ami_available_streams - 2;
	TP_LOG_DEBUG_ID("Reduced merge arity due to AMI restrictions.");
      }
    }
  }
 TP_LOG_DEBUG_ID("AMI_generalized_partition_and_merge(): merge arity = "<< merge_arity);
  if (merge_arity < 2) {
    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
  }
  
  
  //#define MINIMIZE_INITIAL_SUBSTREAM_LENGTH
#ifdef MINIMIZE_INITIAL_SUBSTREAM_LENGTH
  
  // Make the substreams as small as possible without increasing the
  // height of the merge tree.
  {
    // The tree height is the ceiling of the log base merge_arity
    // of the number of original substreams.
    
    double tree_height = log((double)nb_orig_substr)/ log((double)merge_arity);
    tp_assert(tree_height > 0, "Negative or zero tree height!");
    
    tree_height = ceil(tree_height);
    
    // See how many substreams we could possibly fit in the tree
    // without increasing the height.
    double max_original_substreams = pow((double)merge_arity, tree_height);
    tp_assert(max_original_substreams >= nb_orig_substr,
	      "Number of permitted substreams was reduced.");

    // How big will such substreams be?
    double new_sz_original_substream = ceil((double)len /
					    max_original_substreams);
    tp_assert(new_sz_original_substream <= sz_orig_substr,
	      "Size of original streams increased.");
    
    sz_orig_substr = (size_t)new_sz_original_substream;
   TP_LOG_DEBUG_ID("Memory constraints set original substreams = " << nb_orig_substr);
    
    nb_orig_substr = (len + sz_orig_substr - 1) / sz_orig_substr;
   TP_LOG_DEBUG_ID("Tree height constraints set original substreams = " << nb_orig_substr);
  }                
#endif // MINIMIZE_INITIAL_SUBSTREAM_LENGTH

    
  // Create a temporary stream, then iterate through the substreams,
  // processing each one and writing it to the corresponding substream
  // of the temporary stream.
  initial_tmp_stream = new AMI_STREAM<T>;
  mm_stream = new T[(TPIE_OS_SIZE_T)sz_orig_substr];
  tp_assert(mm_stream != NULL, "Misjudged available main memory.");
  if (mm_stream == NULL) {
    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
  }
  
  instream->seek(0);
  assert(ae == AMI_ERROR_NO_ERROR);

  tp_assert(nb_orig_substr * sz_orig_substr - len < sz_orig_substr,
	    "Total substream length too long or too many.");
  tp_assert(len - (nb_orig_substr - 1) * sz_orig_substr <= sz_orig_substr,
	    "Total substream length too short or too few.");        
  
  for (ii = 0; ii++ < nb_orig_substr; ) {
    
    TPIE_OS_OFFSET mm_len;
    if (ii == nb_orig_substr) {
      mm_len = len % sz_orig_substr;
      // If it is an exact multiple, then the mod will come out 0,
      // which is wrong.
      if (!mm_len) {
	mm_len = sz_orig_substr;
      }
    } else {
      mm_len = sz_orig_substr;
    }
#if DEBUG_ASSERTIONS
    TPIE_OS_OFFSET mm_len_bak = mm_len;
#endif
    
    // Read a memory load out of the input stream.
    ae = instream->read_array(mm_stream, &mm_len);
    if (ae != AMI_ERROR_NO_ERROR) {
      return ae;
    }
    tp_assert(mm_len == mm_len_bak,
	      "Did not read the requested number of objects." <<
	      "\n\tmm_len = " << mm_len <<
	      "\n\tmm_len_bak = " << mm_len_bak << '.');
    
    // Solve in main memory. We know it fits, so cast to TPIE_OS_SIZE_T
    m_obj->main_mem_operate(mm_stream, (TPIE_OS_SIZE_T)mm_len);
    
    // Write the result out to the temporary stream.
    ae = initial_tmp_stream->write_array(mm_stream, mm_len);
    if (ae != AMI_ERROR_NO_ERROR) {
      return ae;
    }            
  } //for
  delete [] mm_stream;

  
  // Make sure the total length of the temporary stream is the same as
  // the total length of the original input stream.
  tp_assert(instream->stream_len() == initial_tmp_stream->stream_len(),
	    "Stream lengths do not match:" <<
	    "\n\tinstream->stream_len() = " << instream->stream_len() <<
	    "\n\tinitial_tmp_stream->stream_len() = " <<
	    initial_tmp_stream->stream_len() << ".\n");
  
  // Set up the loop invariants for the first iteration of hte main
  // loop.
  current_input = initial_tmp_stream;
  current_substream_len = sz_orig_substr;
  
  // Pointers to the substreams that will be merged.
  AMI_STREAM<T>* *the_substreams = new AMI_STREAM<T>*[merge_arity];
  
  //Monitoring prints.
 TP_LOG_DEBUG_ID("Number of runs from run formation is "
				 << nb_orig_substr);
 TP_LOG_DEBUG_ID("Merge arity is " << merge_arity);
  
  
  k = 0;
  // The main loop.  At the outermost level we are looping over levels
  // of the merge tree.  Typically this will be very small, e.g. 1-3.
  for( ; current_substream_len < (size_t)len;
       current_substream_len *= merge_arity) {
    
    // The number of substreams to be processed at this level.
    arity_t substream_count;
    
    // Set up to process a given level.
    tp_assert(len == current_input->stream_len(),
	      "Current level stream not same length as input." <<
	      "\n\tlen = " << len <<
	      "\n\tcurrent_input->stream_len() = " <<
	      current_input->stream_len() << ".\n");
    
    // Do we have enough main memory to merge all the substreams on
    // the current level into the output stream?  If so, then we will
    // do so, if not then we need an additional level of iteration to
    // process the substreams in groups.
    substream_count = (arity_t)((len + current_substream_len - 1) /
      current_substream_len);
    
    if (substream_count <= merge_arity) {
      
     TP_LOG_DEBUG_ID("Merging substreams directly to the output stream.");
      
      // Create all the substreams
      for (sub_start = 0, ii = 0 ;
	   ii < substream_count;
	   sub_start += current_substream_len, ii++) {
	
	sub_end = sub_start + current_substream_len - 1;
	if (sub_end >= len) {
	  sub_end = len - 1;
	}
	current_input->new_substream(AMI_READ_STREAM, sub_start, sub_end,
				     (AMI_stream_base<T> **)
				     (the_substreams + ii));
	// The substreams are read-once.
	the_substreams[ii]->persist(PERSIST_READ_ONCE);
      }               
      
      tp_assert(((int) sub_start >= (int) len) &&
		((int) sub_start < (int) len + (int) current_substream_len),
		"Loop ended in wrong location.");
      
      // Fool the OS into unmapping the current block of the input
      // stream so that blocks of the substreams can be mapped in
      // without overlapping it.  This is needed for correct execution
      // on HP-UX.
      //this needs to be cleaned up..Laura
      current_input->seek(0);
      assert(ae == AMI_ERROR_NO_ERROR);

      // Merge them into the output stream.
      ae = AMI_generalized_single_merge(the_substreams, substream_count, outstream, m_obj);
      if (ae != AMI_ERROR_NO_ERROR) {
	return ae;
      }
      // Delete the substreams.
      for (ii = 0; ii < substream_count; ii++) {
	delete the_substreams[ii];
      }
      // And the current input, which is an intermediate stream of
      // some kind.
      delete current_input;
      
    } else {
      
      //substream_count  is >  merge_arity
     TP_LOG_DEBUG_ID("Merging substreams to an intermediate stream.");
      
      // Create the next intermediate stream.
      intermediate_tmp_stream = new AMI_STREAM<T>;
      
      // Fool the OS into unmapping the current block of the input
      // stream so that blocks of the substreams can be mapped in
      // without overlapping it.  This is needed for correct execution
      // on HU-UX.
       //this needs to be cleaned up..Laura
      current_input->seek(0);
      assert(ae == AMI_ERROR_NO_ERROR);

      // Loop through the substreams of the current stream, merging as
      // many as we can at a time until all are done with.
      for (sub_start = 0, ii = 0, jj = 0;
	   ii < substream_count;
	   sub_start += current_substream_len, ii++, jj++) {
	
	sub_end = sub_start + current_substream_len - 1;
	if (sub_end >= len) {
	  sub_end = len - 1;
	}
	current_input->new_substream(AMI_READ_STREAM, sub_start, sub_end,
				     (AMI_stream_base<T> **)
				     (the_substreams + jj));
	// The substreams are read-once.
	the_substreams[jj]->persist(PERSIST_READ_ONCE);
                    
	// If we've got all we can handle or we've seen them all, then
	// merge them.
	if ((jj >= (int) merge_arity - 1) || (ii == substream_count - 1)) {
	  
	  tp_assert(jj <= (int) merge_arity - 1,
		    "Index got too large.");
#if DEBUG_ASSERTIONS
	  // Check the lengths before the merge.
	  TPIE_OS_OFFSET sz_output, sz_output_after_merge;
	  TPIE_OS_OFFSET sz_substream_total;
	  
	  {
	    unsigned int kk;
	    
	    sz_output = intermediate_tmp_stream->stream_len();
	    sz_substream_total = 0;
	    
	    for (kk = jj+1; kk--; ) {
	      sz_substream_total += the_substreams[kk]->stream_len();
	    }                          
	    
	  }
#endif 
	  
	  // This should append to the stream, since
	  // AMI_generalized_single_merge() does not rewind the output before
	  // merging.
	  ae = AMI_generalized_single_merge(the_substreams, jj+1,
				intermediate_tmp_stream, m_obj);
	  if (ae != AMI_ERROR_NO_ERROR) {
	    return ae;
	  }
	  
#if DEBUG_ASSERTIONS
	  // Verify the total lengths after the merge.
	  sz_output_after_merge = intermediate_tmp_stream->stream_len();
	  tp_assert(sz_output_after_merge - sz_output ==
		    sz_substream_total,
		    "Stream lengths do not add up: " <<
		    sz_output_after_merge - sz_output <<
		    " written when " <<
		    sz_substream_total <<
		    " were to have been read.");
                                  
#endif 
	  
	  // Delete the substreams.  jj is currently the index of the
	  // largest, so we want to bump it up before the idiomatic
	  // loop.
	  for (jj++; jj--; ) {
	    delete the_substreams[jj];
	  }
	  
	  // Now jj should be -1 so that it gets bumped back up to 0
	  // before the next iteration of the outer loop.
	  tp_assert((jj == -1), "Index not reduced to -1.");
	  
	} // if                
      } //for
      
      // Get rid of the current input stream and use the next one.
      delete current_input;
      current_input = intermediate_tmp_stream;
    }
    
    k++;
    
  }

  //Monitoring prints.
 TP_LOG_DEBUG_ID("Number of passes incl run formation is " << k+1);
  
  delete [] the_substreams;
  return AMI_ERROR_NO_ERROR;

}

#endif
