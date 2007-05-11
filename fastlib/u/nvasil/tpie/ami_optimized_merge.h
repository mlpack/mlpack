//
// File: ami_optimized_merge.h 
// Author: Rakesh Barve  <rbarve@cs.duke.edu>
//
//cleaned up: laura (tried to..) TO DO: the 3 polymorphs of
//AMI_partition_and_merge() have each 1100 lines of code and are
//almost identical; similarly, the 3 polymorphs of AMI_single_merge()
//differ in one line..
//  
// TO DO: (jan) Check whether all new's are matched by corresponding
//              delete's (especially before prematurely "return"ing).
//
// Function: AMI_partition_and_merge() was modified from Darren's
// original version, so as to ensure "sequential access."  The
// function AMI_single_merge(), which uses a merge management object
// and a priority queue class to carry out internal memory merging
// computation, now has a "pure C" alternative that seems to perform
// better by a huge margin: This function is called AMI_single_merge()
// (a polymorph, without merge management object) and is based on a
// simple heap data structure straight out of CLR (Introduction to
// ALgorithms) in mergeheap.h There is also a merge using
// replacement selection based run formation.  There is also a
// provision to use a run formation that uses a quicksort using only
// keys of the items; there is a provision to to use templated heaps
// to implement the merge.

// $Id: ami_optimized_merge.h,v 1.59 2005/07/07 20:43:49 adanner Exp $

// TO DO: substream_count setting; don't depend on current_stream_len

//COMMENT REGARDING BTE_IMP_USER_DEFINED: USER_DEFINED is what is
//currently the name for STRIPED_BTE. As of now, STRIPED_BTE is not
//part of TPIE distribution. Once it becomes partof TPIE distribution
//the BTE_IMP_USER_DEFINED flag will begin to be used.

#ifndef _AMI_OPTIMIZED_MERGE_H
#define _AMI_OPTIMIZED_MERGE_H

// Get definitions for working with Unix and Windows
#include <portability.h>

// For log() and such as needed to compute tree heights.
#include <math.h>

#include <assert.h>
#include <fstream>

#include <ami_stream.h>
#include <mergeheap.h>		//For templated heaps
#include <quicksort.h>		//For templated qsort_items
#include <tpie_tempnam.h>

typedef int AMI_merge_flag;
typedef int arity_t;

//enable debugging messages in AMI_partition_and_merge(..)
// #define XXX TP_LOG_DEBUG_ID("AMI_partition_and_merge_stream");
#define XXX

//------------------------------------------------------------
// FUNCTIONS DEFINED IN THIS MODULE
//------------------------------------------------------------

//These are polymorphs of AMI_single_merge in ami_merge.h; merge input
//streams using a 'hardwired' heap, without using a merge-management
//object, but:

//using < operator
template < class T >
AMI_err AMI_single_merge (AMI_STREAM < T > **, arity_t,
			  AMI_STREAM < T > *);


// Comment: (jan) Do not use this version anymore

// //do not use <,  use specified comparison function

// End Comment.


//make use of the explicit knowledge of the key of the user-defined
//records
template < class T, class KEY >
AMI_err AMI_single_merge (AMI_STREAM < T > **, arity_t,
			  AMI_STREAM < T > *, int, KEY);

//These are polymorphs of AMI_merge in ami_merge.h, each corresponding
//to one of AMI_single_merge's polymorphs defined above; merge <arity>
//streams using a merge management object and write result into
//<outstream>; it is assumed that the available memory can fit the
//<arity> streams, the output stream and also the space required by
//the merge management object;

template < class T >
AMI_err AMI_merge (AMI_STREAM < T > **, arity_t, AMI_STREAM < T > *);


template < class T, class KEY >
AMI_err AMI_merge (AMI_STREAM < T > **, arity_t,
		   AMI_STREAM < T > *, int, KEY);

//These are polymorphs of AMI_partition_and_merge in
//ami_merge.h;divide the input stream in substreams, merge each
//substream recursively, and merge them together using one of
//AMI_single_merge() polymorphs defined above;

template < class T >
AMI_err AMI_partition_and_merge (AMI_STREAM < T > *instream,
				 AMI_STREAM < T > *outstream);

template < class T, class KEY >
AMI_err AMI_partition_and_merge (AMI_STREAM < T > *instream,
				 AMI_STREAM < T > *outstream,
				 int keyoffset, KEY dummykey);

//------------------------------------------------------------
//static classes functions

//class describing a run formation item
//static template<class KEY> class run_formation_item;

template < class T >
static size_t
count_stream_overhead (AMI_STREAM < T > **instreams, arity_t arity);

template < class T, class KEY >
static AMI_err
Run_Formation_Algo_R_Key (AMI_STREAM < T > *, arity_t, AMI_STREAM < T > **,
			  char *, size_t, int *, int **, int, int, int,
			  KEY);

template < class T, class KEY >
static AMI_err
AMI_replacement_selection_and_merge_Key (AMI_STREAM < T > *instream,
					 AMI_STREAM < T > *outstream,
					 int keyoffset, KEY dummykey);

static inline void
stream_name_generator (char *prepre, char *pre, int id, char *dest);
//------------------------------------------------------------

//------------------------------------------------------------
//class describing a run formation item
template < class KEY > class run_formation_item {
public:
    KEY Key;
    unsigned int RecordPtr;
    unsigned int Loser;
    short RunNumber;
    unsigned int ParentExt;
    unsigned int ParentInt;

public:
    friend int operator == (const run_formation_item & x,
			    const run_formation_item & y)
	{ return (x.Key == y.Key);};

    friend int operator != (const run_formation_item & x,
			    const run_formation_item & y) {
	return (x.Key != y.Key);
    };

    friend int operator <= (const run_formation_item & x,
			    const run_formation_item & y) {
	return (x.Key <= y.Key);
    };

    friend int operator >= (const run_formation_item & x,
			    const run_formation_item & y) {
	return (x.Key >= y.Key);
    };

    friend int operator < (const run_formation_item & x,
			   const run_formation_item & y) {
	return (x.Key < y.Key);
    };

    friend int operator > (const run_formation_item & x,
			   const run_formation_item & y) {
	return (x.Key > y.Key);
    };

};

//------------------------------------------------------------
//This is polymorph to AMI_single_merge in ami_merge.h; merge input
//streams using a 'hardwired' heap, without using a merge-management
//object
//------------------------------------------------------------
template < class T >
AMI_err AMI_single_merge (AMI_STREAM < T > **instreams, arity_t arity,
			  AMI_STREAM < T > *outstream)
{
    unsigned int i, j;
    AMI_err ami_err;
    T merge_out;

    //the mergeheap
    class merge_heap_element < T > *K_Array =
	new merge_heap_element<T>[arity + 1];

    //Pointers to current leading elements of streams
    T* *in_objects = new T*[arity + 1];

    //The number of actual heap elements at any time: can change even
    //after the merge begins because whenever some stream gets
    //completely depleted, heapsize decremnents by one.
    int heapsize_H;

    // Rewind and read the first item from every stream.
    j = 1;
    for (i = 0; i < arity; i++) {

	if ((ami_err = instreams[i]->seek (0)) != AMI_ERROR_NO_ERROR) {
	    delete[] in_objects;
	    return ami_err;
	}
	if ((ami_err = instreams[i]->read_item (&(in_objects[i]))) !=
	    AMI_ERROR_NO_ERROR) {
	    if (ami_err == AMI_ERROR_END_OF_STREAM) {
		in_objects[i] = NULL;
	    } else {
		delete[] in_objects;
		return ami_err;
	    }
	} else {
	    //read_item succesful: Set the taken flags to 0 before we call
	    //intialize()
	    K_Array[j].key = *in_objects[i];
	    K_Array[j].run_id = i;
	    j++;
	}
    }

    //build a heap from the smallest items of each stream
    unsigned int NonEmptyRuns = j - 1;

    merge_heap < T > Main_Merge_Heap (K_Array, NonEmptyRuns);

    while (Main_Merge_Heap.sizeofheap ()) {
	i = Main_Merge_Heap.get_min_run_id ();
	if ((ami_err = outstream->write_item (*in_objects[i]))
	    != AMI_ERROR_NO_ERROR) {
	    delete[] in_objects;
	    return ami_err;
	}
	if ((ami_err = instreams[i]->read_item (&(in_objects[i])))
	    != AMI_ERROR_NO_ERROR) {
	    if (ami_err != AMI_ERROR_END_OF_STREAM) {
		delete[] in_objects;
		return ami_err;
	    }
	}
	if (ami_err == AMI_ERROR_END_OF_STREAM) {
	    Main_Merge_Heap.delete_min_and_insert ((T *) NULL);
	} else {
	    Main_Merge_Heap.delete_min_and_insert (in_objects[i]);
	}
    }				//while

    return AMI_ERROR_NO_ERROR;
}


//------------------------------------------------------------
//This is a polymorph of AMI_single_merge in ami_merge.h; merge input
//streams using a 'hardwired' heap, without using a merge-management
//object; it makes use of the explicit knowledge of the key of the
//user-defined records
//------------------------------------------------------------
template < class T, class KEY >
AMI_err
AMI_single_merge (AMI_STREAM < T > **instreams, arity_t arity,
		  AMI_STREAM < T > *outstream, int keyoffset, KEY dummykey)
{
    unsigned int i, j;
    AMI_err ami_err;
    T merge_out;

/*    //The number of actual heap elements at any time: can change even
    //after the merge begins because whenever some stream gets completely
    //depleted, heapsize decremnents by one.
    int heapsize_H;
*/
    //the mergeheap
    class merge_heap_element < KEY > *K_Array =
	new merge_heap_element<KEY>[arity + 1];

    //Pointers to current leading elements of streams
    T* *in_objects = new T*[arity + 1];

    // Rewind and read the first item from every stream.
    j = 1;
    for (i = 0; i < (int) arity; i++) {

	if ((ami_err = instreams[i]->seek (0)) != AMI_ERROR_NO_ERROR) {
	    delete[] in_objects;
	    return ami_err;
	}
	if ((ami_err = instreams[i]->read_item (&(in_objects[i]))) !=
	    AMI_ERROR_NO_ERROR) {
	    if (ami_err == AMI_ERROR_END_OF_STREAM) {
		in_objects[i] = NULL;
	    } else {
		delete[] in_objects;
		return ami_err;
	    }
	} else {
	    // Set the taken flags to 0 before we call intialize()
	    K_Array[j].key = *((KEY*)((char *) in_objects[i] + keyoffset));
	    K_Array[j].run_id = i;
	    j++;
	}
    }

    //build a heap from the smallest items of each stream
    unsigned int NonEmptyRuns = j - 1;

    merge_heap < KEY > Main_Merge_Heap (K_Array, NonEmptyRuns);

    while (Main_Merge_Heap.sizeofheap ()) {

	i = Main_Merge_Heap.get_min_run_id ();
	if ((ami_err = outstream->write_item (*in_objects[i]))
	    != AMI_ERROR_NO_ERROR) {
	    delete[] in_objects;
	    return ami_err;
	}
	if ((ami_err = instreams[i]->read_item (&(in_objects[i])))
	    != AMI_ERROR_NO_ERROR) {
	    if (ami_err != AMI_ERROR_END_OF_STREAM) {
		delete[] in_objects;
		return ami_err;
	    }
	}
	if (ami_err == AMI_ERROR_END_OF_STREAM) {
	    Main_Merge_Heap.delete_min_and_insert ((KEY *) NULL);
	} else {
	    Main_Merge_Heap.delete_min_and_insert 
		((KEY *) ((char *) in_objects[i] + keyoffset));
	}
    }  //while

    return AMI_ERROR_NO_ERROR;
}

//------------------------------------------------------------ 
//Iterate through the streams, finding out how much additional memory
//each stream will need in the worst case (the streams are in memory,
//but their memory usage could be smaller then the maximum one; one
//scenario is when the streams have been loaded from disk with no
//subsequent read_item/write_item operation, in which case their
//current memory usage is just the header block); count also the
//output stream
//------------------------------------------------------------
template < class T >
size_t
count_stream_overhead (AMI_STREAM < T > **instreams, arity_t arity)
{
    size_t sz_stream, sz_needed = 0;

    for (unsigned int ii = 0; ii < arity + 1; ii++) {
	instreams[ii]->main_memory_usage (&sz_stream,
					  MM_STREAM_USAGE_MAXIMUM);
	sz_needed += sz_stream;
	instreams[ii]->main_memory_usage (&sz_stream,
					  MM_STREAM_USAGE_CURRENT);
	sz_needed -= sz_stream;
    }
    return sz_needed;
}

//------------------------------------------------------------
//These are polymorphs of AMI_merge in ami_merge.h, each corresponding
//to one of AMI_single_merge's polymorphs defined above; merge <arity>
//streams using a merge management object and write result into
//<outstream>; it is assumed that the available memory can fit the
//<arity> streams, the output stream and also the space required by
//the merge management object;

//------------------------------------------------------------
template < class T >
AMI_err
AMI_merge (AMI_STREAM < T > **instreams, arity_t arity,
	   AMI_STREAM < T > *outstream)
{
    size_t sz_avail;
    size_t sz_needed;

    // How much main memory is available?
    sz_avail = MM_manager.memory_available ();

    //make sure all streams fit in available memory
    sz_needed = count_stream_overhead (instreams, arity);
    if (sz_needed >= sz_avail) {
	TP_LOG_FATAL_ID ("Insufficent main memory to perform a merge.");
	return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
    }
    // assert (sz_needed < sz_avail); just checked this.. dh

    //should count the space overhead used by merge..merge should
    //implement a function which returns it; for the moment just rely on
    //the merge routine that it returns an error
    //(AMI_ERROR_INSUFFICIENT_MEMORY) if there is n ot enough memory;

    return AMI_single_merge (instreams, arity, outstream);
}


//------------------------------------------------------------
template < class T, class KEY >
AMI_err
AMI_merge (AMI_STREAM < T > **instreams, arity_t arity,
	   AMI_STREAM < T > *outstream, int keyoffset, KEY dummy)
{
    size_t sz_avail;
    size_t sz_needed;

    // How much main memory is available?
    sz_avail = MM_manager.memory_available ();

    //make sure all streams fit in available memory
    sz_needed = count_stream_overhead (instreams, arity);
    if (sz_needed >= sz_avail) {
	TP_LOG_FATAL_ID ("Insuficent main memory to perform a merge.");
	return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
    }
    // assert (sz_needed < sz_avail); just checked this .. dh

    //should count the space overhead used by merge..merge should
    //implement a function which returns it; for the moment just rely on
    //the merge routine that it returns an error
    //(AMI_ERROR_INSUFFICIENT_MEMORY) if there is n ot enough memory;

    return AMI_single_merge (instreams, arity, keyoffset, dummy);
}

//------------------------------------------------------------
static inline void
stream_name_generator (char *prepre, char *pre, int id, char *dest)
{
    char tmparray[5];

    strcpy (dest, prepre);
    strcat (dest, pre);
    sprintf (tmparray, "%d", id);
    strcat (dest, tmparray);
}

//------------------------------------------------------------
//This is a polymorph of AMI_partition_and_merge in ami_merge.h;divide
//the input stream in substreams, merge each substream recursively,
//and merge them together using AMI_single_merge(AMI_STREAM<T> **,
//arity_t , AMI_STREAM<T> *);
//------------------------------------------------------------
template < class T >
AMI_err
AMI_partition_and_merge (AMI_STREAM < T > *instream,
			 AMI_STREAM < T > *outstream)
{
    AMI_err ae;
    TPIE_OS_OFFSET len;
    size_t sz_avail, sz_stream;
    size_t sz_substream;

    unsigned int ii, jj, kk;
    int ii_streams;

    char *working_disk;

   TP_LOG_DEBUG_ID ("AMI_partition_and_merge_stream START");

    // Figure out how much memory we've got to work with.

    sz_avail = MM_manager.memory_available ();

    //Conservatively assume that the memory for buffers for 
    //the two streams is unallocated; so we need to subtract.
    if ((ae = instream->main_memory_usage (&sz_stream,
					   MM_STREAM_USAGE_MAXIMUM)) !=
	AMI_ERROR_NO_ERROR) {
	TP_LOG_DEBUG_ID ("memory error");
	return ae;
    }

    if ((ae = instream->main_memory_usage (&sz_substream,
					   MM_STREAM_USAGE_OVERHEAD)) !=
	AMI_ERROR_NO_ERROR) {

	TP_LOG_DEBUG_ID ("memory error");
	return ae;
    }
    sz_avail -= 2 * sz_stream;

    working_disk = tpie_tempnam("AMI");

    // If the whole input can fit in main memory then just call
    // AMI_main_mem_merge() to deal with it by loading it once and
    // processing it.

    len = instream->stream_len ();
    instream->seek (0);

    if ((len * sizeof (T)) <= sz_avail) {

	T *next_item;
	T *mm_stream = new T[len];

	for (int i = 0; i < len; i++) {
	    if ((ae = instream->read_item (&next_item)) != AMI_ERROR_NO_ERROR) {
		TP_LOG_DEBUG_ID ("read error");
		return ae;
	    }
	    mm_stream[i] = *next_item;
	}
	quick_sort_op ((T *) mm_stream, len);

	for (int i = 0; i < len; i++) {
	    if ((ae = outstream->write_item (mm_stream[i]))
		!= AMI_ERROR_NO_ERROR) {
		TP_LOG_DEBUG_ID ("write error");
		if (mm_stream) 
		    delete[] mm_stream;
		return ae;
	    }
	}

	if (mm_stream) {
	    delete[] mm_stream;
	    mm_stream = NULL;
	}

	return AMI_ERROR_NO_ERROR;

    } else {

	// The number of substreams that the original input stream
	// will be split into.
	arity_t original_substreams;

	// The length, in terms of stream objects of type T, of the
	// original substreams of the input stream.  The last one may
	// be shorter than this.

	size_t sz_original_substream;

	// The initial temporary stream, to which substreams of the
	// original input stream are written.

	//RAKESH
	AMI_STREAM < T > **initial_tmp_stream;

	// The number of substreams that can be merged together at once.

	arity_t merge_arity;

	// A pointer to the buffer in main memory to read a memory load into.
	T *mm_stream;

	// Loop variables:

	// The stream being read at the current level.

	//RAKESH
	AMI_STREAM < T > **current_input;

	// The output stream for the current level if it is not outstream.

	//RAKESH
	AMI_STREAM < T > **intermediate_tmp_stream;

	//RAKESH  FIX THIS: Need to generate random strings using
	//tmpname() or something like that.
	char *prefix_name[] = { "_0_", "_1_" };
	char itoa_str[5];

	// The size of substreams of *current_input that are being
	// merged.  The last one may be smaller.  This value should be
	// sz_original_substream * (merge_arity ** k) where k is the
	// number of iterations the loop has gone through.

	//Merge Level
	unsigned int k;

	TPIE_OS_OFFSET sub_start, sub_end;

	// How many substreams will there be?  The main memory
	// available to us is the total amount available, minus what
	// is needed for the input stream and the temporary stream.

//RAKESH
// In our case merge_arity is determined differently than in the original
// implementation of AMI_partition_and_merge since we use several streams
// in each level.
// In our case net main memory required to carry out an R-way merge is
// (R+1)*MM_STREAM_USAGE_MAXIMUM  {R substreams for input runs, 1 stream for output}
// + R*MM_STREAM_USAGE_OVERHEAD   {One stream for each active input run: but while
//                                 the substreams use buffers, streams don't}
// + (R+1)*m_obj->space_usage_per_stream();
//
// The net memory usage for an R-way merge is thus
// R*(sz_stream + sz_substeam + m_obj->space_usage_per_stream()) + sz_stream +
// m_obj->space_usage_per_stream();
//

	//To support a binary merge, need space for max_stream_usage
	//for at least three stream objects.

	if (sz_avail <= 3 * (sz_stream + sz_substream
			     + sizeof (merge_heap_element < T >)) ) {
	   TP_LOG_FATAL_ID
		("Insufficient Memory for AMI_partition_and_merge_stream()");
	    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
	}

	sz_original_substream = (sz_avail) / sizeof (T);

	// Round the original substream length off to an integral
	// number of chunks.  This is for systems like HP-UX that
	// cannot map in overlapping regions.  It is also required for
	// BTE's that are capable of freeing chunks as they are
	// read.

	{
	    size_t sz_chunk_size = instream->chunk_size ();

	    sz_original_substream = sz_chunk_size *
		((sz_original_substream + sz_chunk_size - 1) / sz_chunk_size);
	}

	original_substreams = (len + sz_original_substream - 1) /
	    sz_original_substream;

	// Account for the space that a merge object will use.

	{
	    //Availabe memory for input stream objects is given by 
	    //sz_avail minus the space occupied by output stream objects.
	    size_t sz_avail_during_merge = sz_avail -

		sz_stream - sz_substream;

	    //This conts the per-input stream memory cost.
	    size_t sz_stream_during_merge = sz_stream + sz_substream +
		sizeof (merge_heap_element < T >);

	    //Compute merge arity
	    merge_arity = sz_avail_during_merge / sz_stream_during_merge;

	}

	// Make sure that the AMI is willing to provide us with the
	// number of substreams we want.  It may not be able to due to
	// operating system restrictions, such as on the number of
	// regions that can be mmap()ed in.
	{
	    int ami_available_streams = instream->available_streams ();

	    if (ami_available_streams != -1) {
		if (ami_available_streams <= 5) {
		   TP_LOG_FATAL_ID ("out of streams");
		    return AMI_ERROR_INSUFFICIENT_AVAILABLE_STREAMS;
		}

		if (merge_arity > (arity_t) ami_available_streams - 2) {
		    merge_arity = ami_available_streams - 2;
		   TP_LOG_DEBUG_ID
			("Reduced merge arity due to AMI restrictions.");

		}
	    }
	}

	TP_LOG_DEBUG_ID ("AMI_partition_and_merge(): merge arity = " <<
		      merge_arity );

	if (merge_arity < 2) {

	   TP_LOG_FATAL_ID
		("Insufficient memory for AMI_partition_and_merge_stream()");

	    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
	}
//#define MINIMIZE_INITIAL_SUBSTREAM_LENGTH
#ifdef MINIMIZE_INITIAL_SUBSTREAM_LENGTH
	// Make the substreams as small as possible without increasing
	// the height of the merge tree.
	{
	    // The tree height is the ceiling of the log base merge_arity
	    // of the number of original substreams.

	    double tree_height = log ((double) original_substreams) /
		log ((double) merge_arity);

	    tp_assert (tree_height > 0, "Negative or zero tree height!");

	    tree_height = ceil (tree_height);

	    // See how many substreams we could possibly fit in the
	    // tree without increasing the height.

	    double max_original_substreams = pow ((double) merge_arity,
						  tree_height);

	    tp_assert (max_original_substreams >= original_substreams,
		       "Number of permitted substreams was reduced.");

	    // How big will such substreams be?

	    double new_sz_original_substream = ceil ((double) max  len_original_substreams);

	    tp_assert (new_sz_original_substream <= sz_original_substream,
		       "Size of original streams increased.");

	    sz_original_substream = (size_t) new_sz_original_substream;

	   TP_LOG_DEBUG_ID ("Memory constraints set original substreams = " <<
			  original_substreams << '\n');

	    original_substreams = (len + sz_original_substream - 1) /
		sz_original_substream;

	   TP_LOG_DEBUG_ID ("Tree height constraints set original substreams = "
			  << original_substreams << '\n');
	}

#endif				// MINIMIZE_INITIAL_SUBSTREAM_LENGTH

	// Create a temporary stream, then iterate through the
	// substreams, processing each one and writing it to the
	// corresponding substream of the temporary stream.

	//  Comment: (jan) Use VarArray for ANSI-compliance.

	//  unsigned int run_lengths[2][merge_arity]
	//  [   (original_substreams + merge_arity - 1) / merge_arity];
	//  int Sub_Start[merge_arity];

	//  End Comment.

	VarArray3D<unsigned int> 
	    run_lengths(2, merge_arity,
			(original_substreams + merge_arity - 1) / merge_arity);

	VarArray1D<int> Sub_Start(merge_arity);

	//  Comment: (jan) initialization is done by the VarArray constructor.

	// memset ((void *) run_lengths, 0,
	// 2 * merge_arity * ((original_substreams + merge_arity - 1) /
	// merge_arity) * sizeof (unsigned int));

	//  End Comment.

	initial_tmp_stream = new AMI_STREAM<T> *[merge_arity];
	mm_stream = new T[sz_original_substream];

	tp_assert (mm_stream != NULL, "Misjudged available main memory.");

	if (mm_stream == NULL) {
	   TP_LOG_FATAL_ID ("internal error");
	    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
	}

	instream->seek (0);

	tp_assert (original_substreams * sz_original_substream - len <
		   sz_original_substream,
		   "Total substream length too long or too many.");

	tp_assert (len - (original_substreams - 1) * sz_original_substream <=
		   sz_original_substream,
		   "Total substream length too short or too few.");

//RAKESH
	size_t check_size = 0;
	int current_stream = merge_arity - 1;

	int runs_in_current_stream = 0;
	int *desired_runs_in_stream = new int[merge_arity];
	char new_stream_name[BTE_STREAM_PATH_NAME_LEN];

	//For the first stream:
	for (ii_streams = 0; ii_streams < merge_arity; ii_streams++) {

	    //Figure out how many runs go in each one of merge_arity streams?
	    // If there are 12 runs to be distributed among 5 streams, the first 
	    //three get 2 and the last two  get 3 runs 

	    if (ii_streams <
		(merge_arity -
		 (original_substreams %
		  merge_arity))) desired_runs_in_stream[ii_streams] =
				     original_substreams / merge_arity;

	    else
		desired_runs_in_stream[ii_streams] =
		    (original_substreams + merge_arity - 1) / merge_arity;
	}

#ifndef BTE_IMP_USER_DEFINED

//    new_name_from_prefix(prefix_name[0],current_stream, new_stream_name);

	//The assumption here is that working_disk is the name of the specific 
	//directory in which the temporary/intermediate streams will be made.
	//By default, I think we shd 

	stream_name_generator (working_disk,
			       prefix_name[0],
			       current_stream, new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
	stream_name_generator ("",
			       prefix_name[0],
			       current_stream, new_stream_name);

#endif

	initial_tmp_stream[current_stream] =
	    new AMI_STREAM < T > (new_stream_name);
	initial_tmp_stream[current_stream]->persist (PERSIST_PERSISTENT);

	ii = 0;
	while (ii < original_substreams) {
	    TPIE_OS_OFFSET mm_len;

	    // Make sure that the current_stream is supposed to get a run

	    if (desired_runs_in_stream[current_stream] >
		runs_in_current_stream) {
		if (ii == original_substreams - 1) {
		    mm_len = len % sz_original_substream;

		    // If it is an exact multiple, then the mod will come
		    // out 0, which is wrong.

		    if (!mm_len) {
			mm_len = sz_original_substream;
		    }
		} else {
		    mm_len = sz_original_substream;
		}

#if DEBUG_ASSERTIONS
		TPIE_OS_OFFSET mm_len_bak = mm_len;
#endif

		// Read a memory load out of the input stream one item at a time,
		// fill up the key array at the same time.
		{
		    T *next_item;

		    for (int i = 0; i < mm_len; i++) {
			if ((ae = instream->read_item (&next_item)) !=
			    AMI_ERROR_NO_ERROR) {
			   TP_LOG_DEBUG_ID ("read error");
			    return ae;
			}
			mm_stream[i] = *next_item;
		    }

		    //Sort the array.
		    quick_sort_op ((T *) mm_stream, mm_len);

		    for (int i = 0; i < mm_len; i++) {
			if (
			    (ae =
			     initial_tmp_stream[current_stream]->write_item
			     (mm_stream[i])) != AMI_ERROR_NO_ERROR) {
			   TP_LOG_DEBUG_ID ("write error");
			    return ae;
			}

		    }

		    run_lengths(0, current_stream, runs_in_current_stream) =
			mm_len;

		}

		runs_in_current_stream++;
		ii++;

	    }
//RAKESH        
	    if (runs_in_current_stream ==
		desired_runs_in_stream[current_stream]) {

		check_size +=
		    initial_tmp_stream[current_stream]->stream_len ();

		// We do not want old streams hanging around
		// occuping memory. We know how to get the streams
		// since we can generate their names
		if (initial_tmp_stream[current_stream]) {

		    delete initial_tmp_stream[current_stream];

		    initial_tmp_stream[current_stream] = NULL;

		}

		if (check_size < instream->stream_len ()) {

		    current_stream = (current_stream + merge_arity - 1)
			% merge_arity;

#ifndef BTE_IMP_USER_DEFINED
		    //    new_name_from_prefix(prefix_name[0],current_stream, new_stream_name);

		    stream_name_generator (working_disk,
					   prefix_name[0],
					   current_stream, new_stream_name);

#endif

#ifdef BTE_IMP_USER_DEFINED
		    stream_name_generator ("",
					   prefix_name[0],
					   current_stream, new_stream_name);
#endif

		    initial_tmp_stream[current_stream] =
			new AMI_STREAM < T > (new_stream_name);

		    initial_tmp_stream[current_stream]->persist
			(PERSIST_PERSISTENT);

		    // Number of runs packed into 
		    // the stream just constructed now

		    runs_in_current_stream = 0;
		}
	    }

	}

	if (initial_tmp_stream[current_stream]) {
	    delete initial_tmp_stream[current_stream];

	    initial_tmp_stream[current_stream] = NULL;
	}

	if (mm_stream) {
	    delete[]mm_stream;
	    mm_stream = NULL;
	}
	// Make sure the total length of the temporary stream is the
	// same as the total length of the original input stream.

	tp_assert (instream->stream_len () == check_size,
		   "Stream lengths do not match:" <<
		   "\n\tinstream->stream_len() = " << instream->stream_len ()
		   << "\n\tinitial_tmp_stream->stream_len() = " << check_size
		   << ".\n");

	//We now delete the input stream. Note that if instream has
	//its persistence member set to PERSIST_DELETE, instream will
	//be deleted from disk.

	//delete instream;

	// Set up the loop invariants for the first iteration of hte
	// main loop.

	current_input = initial_tmp_stream;

	//Monitoring prints.

	TP_LOG_DEBUG_ID ("Number of runs from run formation is " <<
		      original_substreams );
	TP_LOG_DEBUG_ID ("Merge arity is " << merge_arity );

	// Pointers to the substreams that will be merged.
//RAKESH        
	AMI_STREAM < T > **the_substreams =
	    new AMI_STREAM<T>*[merge_arity];

	k = 0;

	// The main loop.  At the outermost level we are looping over
	// levels of the merge tree.  Typically this will be very
	// small, e.g. 1-3.

	T dummykey;		// This is for the last arg to 

	// AMI_single_merge()
	// which necessitated due to type unificatuon problems

	// The number of substreams to be processed at any merge level.
	arity_t substream_count;

	for (substream_count = original_substreams;
	     substream_count > 1;
	     substream_count = (substream_count + merge_arity - 1)
		 / merge_arity) {

	    // Set up to process a given level.
//RAKESH
	    tp_assert (len == check_size,
		       "Current level stream not same length as input." <<
		       "\n\tlen = " << len <<
		       "\n\tcurrent_input->stream_len() = " <<
		       check_size << ".\n");

	    check_size = 0;

	    // Do we have enough main memory to merge all the
	    // substreams on the current level into the output stream?
	    // If so, then we will do so, if not then we need an
	    // additional level of iteration to process the substreams
	    // in groups.

	    if (substream_count <= merge_arity) {

//RAKESH   Open up the substream_count streams in which the
//         the runs input to the current merge level are packed
//         The names of these streams (storing the input runs)
//         can be constructed from  prefix_name[k % 2]

		for (ii = merge_arity - substream_count; ii < merge_arity;
		     ii++) {

#ifndef BTE_IMP_USER_DEFINED

		    stream_name_generator (working_disk,
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);

#endif

#ifdef  BTE_IMP_USER_DEFINED
		    stream_name_generator ("",
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);
#endif

		    current_input[ii] = new AMI_STREAM < T > (new_stream_name);
		    current_input[ii]->persist (PERSIST_DELETE);

		}

		// Merge them into the output stream.

		ae = AMI_single_merge (
		    (current_input + merge_arity -
		     substream_count), substream_count,
		    outstream);

		if (ae != AMI_ERROR_NO_ERROR) {
		   TP_LOG_FATAL_ID ("AMI_single_merge error " <<
				  ae << " returned by  AMI_single_merge()");
		    return ae;
		}
		// Delete the streams input to the above merge.

		for (ii = merge_arity - substream_count;
		     ii < merge_arity; ii++) {

		    if (current_input[ii]) {
			delete current_input[ii];

			current_input[ii] = NULL;
		    }

		}

		if (current_input) {
		    delete[]current_input;
		    current_input = NULL;
		}
		if (the_substreams) {
		    delete[]the_substreams;
		    the_substreams = NULL;
		}

	    } else {

		TP_LOG_DEBUG_ID ("Merging substreams to intermediate streams.");

		// Create the array of merge_arity stream pointers that
		// will each point to a stream containing runs output
		// at the current level k. 

		intermediate_tmp_stream = new AMI_STREAM<T>*[merge_arity];

//RAKESH   Open up the merge_arity streams in which the
//         the runs input to the current merge level are packed
//         The names of these streams (storing the input runs)
//         can be constructed from  prefix_name[k % 2]

		for (ii = 0; ii < merge_arity; ii++) {

//                      new_name_from_prefix(prefix_name[k % 2],(int) ii,
//                                            new_stream_name);

#ifndef BTE_IMP_USER_DEFINED
		    stream_name_generator (working_disk,
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
		    stream_name_generator ("",
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);
#endif

		    current_input[ii] = new AMI_STREAM < T > (new_stream_name);

		    current_input[ii]->persist (PERSIST_DELETE);

		}

		// Fool the OS into unmapping the current block of the
		// input stream so that blocks of the substreams can
		// be mapped in without overlapping it.  This is
		// needed for correct execution on HU-UX.
//RAKESH
//                current_input->seek(0);

		current_stream = merge_arity - 1;

		//For the first stream that we use to pack some    
		//of the output runs of the current merge level k.

//                new_name_from_prefix(prefix_name[(k+1) % 2],0,
//                                            new_stream_name);

#ifndef BTE_IMP_USER_DEFINED

		stream_name_generator (working_disk,
				       prefix_name[(k + 1) % 2],
				       current_stream, new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
		stream_name_generator ("",
				       prefix_name[(k + 1) % 2],
				       current_stream, new_stream_name);
#endif

		intermediate_tmp_stream[current_stream] = new
		    AMI_STREAM < T > (new_stream_name);

		intermediate_tmp_stream[current_stream]->persist
		    (PERSIST_PERSISTENT);

		int remaining_number_of_output_runs =
		    (substream_count + merge_arity - 1) / merge_arity;

		for (ii_streams = 0; ii_streams < merge_arity; ii_streams++) {
		    // If there are 12 runs to be distributed among 5 streams, 
		    // the first three get 2 and the last two  get 3 runs   

		    if (ii_streams <
			(merge_arity -
			 (remaining_number_of_output_runs % merge_arity)))

			desired_runs_in_stream[ii_streams] =
			    remaining_number_of_output_runs / merge_arity;

		    else
			desired_runs_in_stream[ii_streams] =
			    (remaining_number_of_output_runs
			     + merge_arity - 1) / merge_arity;

		    Sub_Start(ii_streams) = 0;

		}

		runs_in_current_stream = 0;
		unsigned int merge_number = 0;

		// Loop through the substreams of the current stream,
		// merging as many as we can at a time until all are
		// done with.
		for (sub_start = 0, ii = 0, jj = 0; ii < substream_count; ii++) {

		    if (run_lengths(k % 2, merge_arity - 1 - jj, merge_number)
			!= 0) {

			sub_start = Sub_Start(merge_arity - 1 - jj);

			sub_end = sub_start +
			    run_lengths(k % 2, merge_arity - 1 -
					       jj, merge_number) - 1;

			Sub_Start(merge_arity - 1 - jj) +=
			    run_lengths(k % 2, merge_arity - 1 -
					       jj, merge_number);

			run_lengths(k % 2, merge_arity - 1 - jj, merge_number)
			    = 0;
		    } else {
			//This weirdness is caused by the way bte substream
			//constructor was designed.

			sub_end = Sub_Start(merge_arity - 1 - jj) - 1;
			sub_start = sub_end + 1;

			ii--;

		    }

		    //Open the new substream
		    current_input[merge_arity - 1 -
				  jj]->new_substream (AMI_READ_STREAM,
						      sub_start, sub_end,
						      (AMI_stream_base < T > **)
						      (the_substreams + jj));

		    // The substreams are read-once.
		    // If we've got all we can handle or we've seen
		    // them all, then merge them.

		    if ((jj >= merge_arity - 1) || (ii == substream_count - 1)) {

			tp_assert (jj <= merge_arity - 1,
				   "Index got too large.");

			//Check if the stream into which runs are cuurently 
			//being packed has got its share of runs. If yes,
			//delete that stream and construct a new stream 
			//appropriately.

			if (desired_runs_in_stream[current_stream] ==
			    runs_in_current_stream) {

			    //Make sure that the deleted stream persists on disk.
			    intermediate_tmp_stream[current_stream]->persist
				(PERSIST_PERSISTENT);

			    delete intermediate_tmp_stream[current_stream];

			    current_stream = (current_stream + merge_arity - 1)
				% merge_arity;

			    // Unless the current level is over, we've to generate 
			    //a new stream for the next set of runs.

			    if (remaining_number_of_output_runs > 0) {

//                        new_name_from_prefix(prefix_name[(k+1) % 2],
//                              current_stream, new_stream_name);

#ifndef BTE_IMP_USER_DEFINED
				stream_name_generator (working_disk,
						       prefix_name[(k + 1) % 2],
						       (int) current_stream,
						       new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
				stream_name_generator ("",
						       prefix_name[(k + 1) % 2],
						       (int) current_stream,
						       new_stream_name);
#endif

				intermediate_tmp_stream[current_stream] = new
				    AMI_STREAM < T > (new_stream_name);

				intermediate_tmp_stream[current_stream]->persist
				    (PERSIST_PERSISTENT);
				runs_in_current_stream = 0;
			    }
			}

			ae = AMI_single_merge (the_substreams,
					       jj + 1,
					       intermediate_tmp_stream
					       [current_stream]);

			if (ae != AMI_ERROR_NO_ERROR) {
			   TP_LOG_DEBUG_ID ("AMI_single_merge error");
			    return ae;
			}

			for (ii_streams = 0; ii_streams < jj + 1; ii_streams++)
			    run_lengths((k + 1) % 2, current_stream,
					runs_in_current_stream) +=
				the_substreams[ii_streams]->stream_len ();

			merge_number++;

			//Decrement the counter corresp to number of runs 
			// still to be formed at current level

			remaining_number_of_output_runs--;

			// Delete input substreams. jj is currently the index
			// of the largest.

			for (ii_streams = 0; ii_streams < jj + 1; ii_streams++) {
			    if (the_substreams[ii_streams]) {
				delete the_substreams[ii_streams];

				the_substreams[ii_streams] = NULL;
			    }
			}

			jj = 0;

//RAKESH                The number of runs in the current_stream
//                      goes up by 1.

			runs_in_current_stream++;

		    } else {
			jj++;
		    }

		}

		if (intermediate_tmp_stream[current_stream]) {
		    delete intermediate_tmp_stream[current_stream];

		    intermediate_tmp_stream[current_stream] = NULL;
		}
		// Get rid of the current input streams and use the ones
		//output at the current level.
//RAKESH

		for (ii = 0; ii < merge_arity; ii++)
		    if (current_input[ii]) {
			delete current_input[ii];
		    }
		if (current_input) {
		    delete[]current_input;
		    current_input = NULL;
		}

		current_input = (AMI_STREAM < T > **)intermediate_tmp_stream;

	    }

	    k++;

	}

	//Monitoring prints.
	TP_LOG_DEBUG_ID ("Number of passes incl run formation is " << k +
		      1 );

	return AMI_ERROR_NO_ERROR;

    }
   TP_LOG_DEBUG_ID ("AMI_partition_and_merge_stream END");
}

//------------------------------------------------------------
//This is a polymorph of AMI_partition_and_merge in ami_merge.h;divide
//the input stream in substreams, merge each substream recursively,
//and merge them together using AMI_single_merge(AMI_STREAM<T> **,
//arity_t , AMI_STREAM<T> *, int , KEY)
//------------------------------------------------------------
template < class T, class KEY >
AMI_err
AMI_partition_and_merge (AMI_STREAM < T > *instream,
			 AMI_STREAM < T > *outstream,
			 int keyoffset, KEY dummykey)
{
    AMI_err ae;
    TPIE_OS_OFFSET len;
    size_t sz_avail, sz_stream;
    size_t sz_substream;

    unsigned int ii, jj;
    int ii_streams;

    char *working_disk;

   TP_LOG_DEBUG_ID ("AMI_partition_and_merge_Key: start");

    // Figure out how much memory we've got to work with.

    sz_avail = MM_manager.memory_available ();

    //Conservatively assume that the memory for buffers for 
    //the two streams is unallocated; so we need to subtract.

    if ((ae = instream->main_memory_usage (&sz_stream,
					   MM_STREAM_USAGE_MAXIMUM)) !=
	AMI_ERROR_NO_ERROR) {
	return ae;
    }

    if ((ae = instream->main_memory_usage (&sz_substream,
					   MM_STREAM_USAGE_OVERHEAD)) !=
	AMI_ERROR_NO_ERROR) {

	return ae;
    }

    sz_avail -= 2 * sz_stream;

    working_disk = tpie_tempnam ("AMI");
    //TP_LOG_DEBUG_ID(working_disk);

    // If the whole input can fit in main memory then just call
    // AMI_main_mem_merge() to deal with it by loading it once and
    // processing it.

    len = instream->stream_len ();
    instream->seek (0);

    if ((len * sizeof (T)) <= sz_avail) {

	if (len * (sizeof (T) * sizeof (qsort_item < KEY >)) > sz_avail)
	    // ie if you have dont have space for separate
	    // keysorting (good cache performance) followed by permuting 
	{

	    T *next_item;

	   TP_LOG_DEBUG_ID ("pre new");
	    T *mm_stream = new T[(TPIE_OS_SIZE_T)len];

	   TP_LOG_DEBUG_ID ("post new");

	    for (int i = 0; i < len; i++) {
		if ((ae = instream->read_item (&next_item)) !=
		    AMI_ERROR_NO_ERROR) return ae;
		mm_stream[i] = *next_item;
	    }

	    quick_sort_op ((T *) mm_stream, (TPIE_OS_SIZE_T)len);

	    for (int i = 0; i < len; i++) {
		if ((ae = outstream->write_item (mm_stream[i]))
		    != AMI_ERROR_NO_ERROR)
		    return ae;
	    }
	   TP_LOG_DEBUG_ID ("pre delete");
	    if (mm_stream) {
		delete[]mm_stream;
		mm_stream = NULL;
	    }
	   TP_LOG_DEBUG_ID ("post delete");
	} else {
	    //Use qsort on keys followed by permuting
	   TP_LOG_DEBUG_ID ("pre new");
	    T *mm_stream = new T[(TPIE_OS_SIZE_T)len];

	    qsort_item < KEY > *qs_array = new qsort_item <KEY>[(TPIE_OS_SIZE_T)len];
	   TP_LOG_DEBUG_ID ("post new");
	    T *next_item;

	    for (int i = 0; i < len; i++) {
		if ((ae = instream->read_item (&next_item)) !=
		    AMI_ERROR_NO_ERROR) return ae;
		mm_stream[i] = *next_item;
		qs_array[i].keyval = *(KEY *) ((char *) next_item + keyoffset);
		qs_array[i].source = i;
	    }

	    quick_sort_op ((qsort_item < KEY > *)qs_array, (TPIE_OS_SIZE_T)len);

	    for (int i = 0; i < len; i++) {
		if (
		    (ae =
		     outstream->write_item (mm_stream[qs_array[i].source])) !=
		    AMI_ERROR_NO_ERROR) return ae;
	    }
	   TP_LOG_DEBUG_ID ("pre delete");
	    if (mm_stream) {
		delete[]mm_stream;
		mm_stream = NULL;
	    }
	    if (qs_array) {
		delete[]qs_array;
		qs_array = NULL;
	    }
	   TP_LOG_DEBUG_ID ("post delete");
	}

	TP_LOG_DEBUG_ID ("AMI_partition_and_merge_Key: done");
	return AMI_ERROR_NO_ERROR;

    } else {

	// The number of substreams that the original input stream
	// will be split into.

	arity_t original_substreams;

	// The length, in terms of stream objects of type T, of the
	// original substreams of the input stream.  The last one may
	// be shorter than this.

	TPIE_OS_OFFSET sz_original_substream;

	// The initial temporary stream, to which substreams of the
	// original input stream are written.

	//RAKESH
	AMI_STREAM < T > **initial_tmp_stream;

	// The number of substreams that can be merged together at once.

	arity_t merge_arity;

	// A pointer to the buffer in main memory to read a memory load into.
	T *mm_stream;

	// Loop variables:

	// The stream being read at the current level.

	//RAKESH
	AMI_STREAM < T > **current_input;

	// The output stream for the current level if it is not outstream.

	//RAKESH
	AMI_STREAM < T > **intermediate_tmp_stream;

	//RAKESH  FIX THIS: Need to generate random strings using
	//tmpname() or something like that.
	char *prefix_name[] = { "_0_", "_1_" };

	// The size of substreams of *current_input that are being
	// merged.  The last one may be smaller.  This value should be
	// sz_original_substream * (merge_arity ** k) where k is the
	// number of iterations the loop has gone through.

	//Merge Level
	unsigned int k;

	TPIE_OS_OFFSET sub_start, sub_end;

	// How many substreams will there be?  The main memory
	// available to us is the total amount available, minus what
	// is needed for the input stream and the temporary stream.

//RAKESH
// In our case merge_arity is determined differently than in the original
// implementation of AMI_partition_and_merge since we use several streams
// in each level.
// In our case net main memory required to carry out an R-way merge is
// (R+1)*MM_STREAM_USAGE_MAXIMUM  {R substreams for input runs, 1 stream for output}
// + R*MM_STREAM_USAGE_OVERHEAD   {One stream for each active input run: but while
//                                 the substreams use buffers, streams don't}
// + (R+1)*m_obj->space_usage_per_stream();
//
// The net memory usage for an R-way merge is thus
// R*(sz_stream + sz_substeam + m_obj->space_usage_per_stream()) + sz_stream +
// m_obj->space_usage_per_stream();
//

	//To support a binary merge, need space for max_stream_usage
	//for at least three stream objects.

	if (sz_avail <= 3 * (sz_stream + sz_substream
			     + sizeof (merge_heap_element < KEY >))
	    //+ sz_stream + sizeof(merge_heap_element<KEY>)
	    ) {

	   TP_LOG_FATAL_ID
		("Insufficient memory in AMI_partition_and_merge_Key()");
	    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
	}

	sz_original_substream =
	    (sz_avail) / (sizeof (T) + sizeof (qsort_item < KEY >));

	// Round the original substream length off to an integral
	// number of chunks.  This is for systems like HP-UX that
	// cannot map in overlapping regions.  It is also required for
	// BTE's that are capable of freeing chunks as they are
	// read.

	{
	    TPIE_OS_OFFSET sz_chunk_size = instream->chunk_size ();

	    sz_original_substream = sz_chunk_size *
		((sz_original_substream + sz_chunk_size - 1) / sz_chunk_size);
	}

	original_substreams = static_cast<arity_t>((len + sz_original_substream - 1) /
	    sz_original_substream);

	// Account for the space that a merge object will use.

	{
	    //Availabe memory for input stream objects is given by 
	    //sz_avail minus the space occupied by output stream objects.
	    TPIE_OS_SIZE_T sz_avail_during_merge = sz_avail -

		sz_stream - sz_substream;

	    //This conts the per-input stream memory cost.
	    TPIE_OS_SIZE_T sz_stream_during_merge = sz_stream + sz_substream +
		sizeof (merge_heap_element < KEY >);

	    //Compute merge arity
	    merge_arity = static_cast<arity_t>(sz_avail_during_merge / sz_stream_during_merge);

	}

	// Make sure that the AMI is willing to provide us with the
	// number of substreams we want.  It may not be able to due to
	// operating system restrictions, such as on the number of
	// regions that can be mmap()ed in.

	{
	    int ami_available_streams = instream->available_streams ();

	    if (ami_available_streams != -1) {
		if (ami_available_streams <= 5) {
		    return AMI_ERROR_INSUFFICIENT_AVAILABLE_STREAMS;
		}

		if (merge_arity > (arity_t) ami_available_streams - 2) {
		    merge_arity = ami_available_streams - 2;
		   TP_LOG_DEBUG_ID
			("Reduced merge arity due to AMI restrictions.");

		}
	    }
	}

	TP_LOG_DEBUG_ID ("AMI_partition_and_merge_Key(): merge arity = " <<
		      merge_arity );

	if (merge_arity < 2) {

	   TP_LOG_FATAL_ID
		("Insufficient memory for AMI_partition_and_merge_Key()");

	    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
	}
//#define MINIMIZE_INITIAL_SUBSTREAM_LENGTH
#ifdef MINIMIZE_INITIAL_SUBSTREAM_LENGTH

	// Make the substreams as small as possible without increasing
	// the height of the merge tree.

	{
	    // The tree height is the ceiling of the log base merge_arity
	    // of the number of original substreams.

	    double tree_height = log ((double) original_substreams) /
		log ((double) merge_arity);

	    tp_assert (tree_height > 0, "Negative or zero tree height!");

	    tree_height = ceil (tree_height);

	    // See how many substreams we could possibly fit in the
	    // tree without increasing the height.

	    double max_original_substreams = pow ((double) merge_arity,
						  tree_height);

	    tp_assert (max_original_substreams >= original_substreams,
		       "Number of permitted substreams was reduced.");

	    // How big will such substreams be?

	    double new_sz_original_substream = ceil ((double) len /

						     max_original_substreams);

	    tp_assert (new_sz_original_substream <= sz_original_substream,
		       "Size of original streams increased.");

	    sz_original_substream = (size_t) new_sz_original_substream;

	   TP_LOG_DEBUG_ID ("Memory constraints set original substreams = " <<
			  original_substreams << '\n');

	    original_substreams = (len + sz_original_substream - 1) /
		sz_original_substream;

	   TP_LOG_DEBUG_ID ("Tree height constraints set original substreams = "
			  << original_substreams << '\n');
	}

#endif				// MINIMIZE_INITIAL_SUBSTREAM_LENGTH

	// Create a temporary stream, then iterate through the
	// substreams, processing each one and writing it to the
	// corresponding substream of the temporary stream.

	//  Comment: (jan) Use VarArray for ANSI-compliance.

	//  unsigned int run_lengths[2][merge_arity]
	//  [   (original_substreams + merge_arity - 1) / merge_arity];
	//  int Sub_Start[merge_arity];

	//  End Comment.

	VarArray3D<TPIE_OS_OFFSET> 
	    run_lengths(2, merge_arity,
			(original_substreams + merge_arity - 1) / merge_arity);

	VarArray1D<TPIE_OS_OFFSET> Sub_Start(merge_arity);

	//  Comment: (jan) initialization is done by the VarArray constructor.

	// memset ((void *) run_lengths, 0,
	// 2 * merge_arity * ((original_substreams + merge_arity - 1) /
	// merge_arity) * sizeof (unsigned int));

	//  End Comment.

	initial_tmp_stream = new AMI_STREAM<T>*[merge_arity];
	TP_LOG_DEBUG_ID ("pre new");
	mm_stream = new T[(TPIE_OS_SIZE_T)sz_original_substream];

	qsort_item < KEY > *qs_array =
	    new qsort_item<KEY>[(TPIE_OS_SIZE_T)sz_original_substream];
	TP_LOG_DEBUG_ID ("post new");

	tp_assert (mm_stream != NULL, "Misjudged available main memory.");

	if (mm_stream == NULL) {

	    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
	}

	instream->seek (0);

	tp_assert (original_substreams * sz_original_substream - len <
		   sz_original_substream,
		   "Total substream length too long or too many.");

	tp_assert (len - (original_substreams - 1) * sz_original_substream <=
		   sz_original_substream,
		   "Total substream length too short or too few.");

//RAKESH
	TPIE_OS_OFFSET check_size = 0;
	int current_stream = merge_arity - 1;

	int runs_in_current_stream = 0;
	int *desired_runs_in_stream = new int[merge_arity];
	char new_stream_name[BTE_STREAM_PATH_NAME_LEN];

	//For the first stream:

	for (ii_streams = 0; ii_streams < (int) merge_arity; ii_streams++) {

	    //Figure out how many runs go in each one of merge_arity streams?
	    // If there are 12 runs to be distributed among 5 streams, the first 
	    //three get 2 and the last two  get 3 runs 

	    if (ii_streams <
		(int) (merge_arity -
		       (original_substreams %
			merge_arity))) desired_runs_in_stream[ii_streams] =
					   original_substreams / merge_arity;

	    else
		desired_runs_in_stream[ii_streams] =
		    (original_substreams + merge_arity - 1) / merge_arity;
	}

#ifndef BTE_IMP_USER_DEFINED

//    new_name_from_prefix(prefix_name[0],current_stream, new_stream_name);

	//The assumption here is that working_disk is the name of the specific 
	//directory in which the temporary/intermediate streams will be made.
	//By default, I think we shd 

	stream_name_generator (working_disk,
			       prefix_name[0],
			       current_stream, new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
	stream_name_generator ("",
			       prefix_name[0],
			       current_stream, new_stream_name);

#endif

	initial_tmp_stream[current_stream] =
	    new AMI_STREAM < T > (new_stream_name);

	initial_tmp_stream[current_stream]->persist (PERSIST_PERSISTENT);

	ii = 0;
	while (ii < original_substreams) {
	    TPIE_OS_SIZE_T mm_len;

	    // Make sure that the current_stream is supposed to get a run

	    if (desired_runs_in_stream[current_stream] >
		runs_in_current_stream) {
		if (ii == original_substreams - 1) {
		    mm_len = static_cast<TPIE_OS_SIZE_T>(len % sz_original_substream);

		    // If it is an exact multiple, then the mod will come
		    // out 0, which is wrong.

		    if (!mm_len) {
			mm_len = static_cast<TPIE_OS_SIZE_T>(sz_original_substream);
		    }
		} else {
		    mm_len = static_cast<TPIE_OS_SIZE_T>(sz_original_substream);
		}

#if DEBUG_ASSERTIONS
		TPIE_OS_OFFSET mm_len_bak = mm_len;
#endif

		// Read a memory load out of the input stream one item at a time,
		// fill up the key array at the same time.

		{
		    T *next_item;

		    for (int i = 0; i < mm_len; i++) {
			if ((ae = instream->read_item (&next_item)) !=
			    AMI_ERROR_NO_ERROR) return ae;
			mm_stream[i] = *next_item;
			qs_array[i].keyval =
			    *(KEY *) ((char *) next_item + keyoffset);
			qs_array[i].source = i;
		    }

		    //Sort the key array.

		    quick_sort_op ((qsort_item < KEY > *)qs_array, mm_len);

		    //Now permute the memoryload as per the sorted key array.

		    for (int i = 0; i < mm_len; i++) {
			if (
			    (ae =
			     initial_tmp_stream[current_stream]->write_item
			     (mm_stream[qs_array[i].source]))
			    != AMI_ERROR_NO_ERROR)
			    return ae;

		    }

		    run_lengths(0, current_stream, runs_in_current_stream) =
			mm_len;

		}

		runs_in_current_stream++;
		ii++;

	    }
//RAKESH        
	    if (runs_in_current_stream ==
		desired_runs_in_stream[current_stream]) {

		check_size +=
		    initial_tmp_stream[current_stream]->stream_len ();

		// We do not want old streams hanging around
		// occuping memory. We know how to get the streams
		// since we can generate their names

		if (initial_tmp_stream[current_stream]) {
		    delete initial_tmp_stream[current_stream];

		    initial_tmp_stream[current_stream] = NULL;
		}

		if ((int) check_size < instream->stream_len ()) {

		    current_stream = (current_stream + merge_arity - 1)
			% merge_arity;

#ifndef BTE_IMP_USER_DEFINED
		    //    new_name_from_prefix(prefix_name[0],current_stream, new_stream_name);

		    stream_name_generator (working_disk,
					   prefix_name[0],
					   current_stream, new_stream_name);

#endif

#ifdef BTE_IMP_USER_DEFINED
		    stream_name_generator ("",
					   prefix_name[0],
					   current_stream, new_stream_name);
#endif

		    initial_tmp_stream[current_stream] =
			new AMI_STREAM < T > (new_stream_name);

		    initial_tmp_stream[current_stream]->persist
			(PERSIST_PERSISTENT);

		    // Number of runs packed into 
		    // the stream just constructed now

		    runs_in_current_stream = 0;
		}
	    }

	}

	if (initial_tmp_stream[current_stream]) {
	    delete initial_tmp_stream[current_stream];

	    initial_tmp_stream[current_stream] = NULL;
	}
	TP_LOG_DEBUG_ID ("pre delete");
	if (mm_stream) {
	    delete[]mm_stream;
	    mm_stream = NULL;
	}
	if (qs_array) {
	    delete[]qs_array;
	    qs_array = NULL;
	}
	TP_LOG_DEBUG_ID ("post delete");

	// Make sure the total length of the temporary stream is the
	// same as the total length of the original input stream.

	tp_assert (instream->stream_len () == check_size,
		   "Stream lengths do not match:" <<
		   "\n\tinstream->stream_len() = " << instream->stream_len ()
		   << "\n\tinitial_tmp_stream->stream_len() = " << check_size
		   << ".\n");

	//We now delete the instream; note that it will be wiped off
	//disk if instream->persistence is set to PERSIST_DELETE
	//delete instream;

	// Set up the loop invariants for the first iteration of hte
	// main loop.

	current_input = initial_tmp_stream;

	// Pointers to the substreams that will be merged.
//RAKESH        
	AMI_STREAM < T > **the_substreams =
	    new AMI_STREAM<T>*[merge_arity];

	k = 0;

	// The main loop.  At the outermost level we are looping over
	// levels of the merge tree.  Typically this will be very
	// small, e.g. 1-3.

	KEY dummykey;		// This is for the last arg to 

	// AMI_partition_and_merge_Key()
	// which necessitated due to type unificatuon problems

	// The number of substreams to be processed at any merge level.
	arity_t substream_count;

	//Monitoring prints.

	TP_LOG_DEBUG_ID ("Number of runs from run formation is " <<
		      original_substreams );
	TP_LOG_DEBUG_ID ("Merge arity is " << merge_arity );

	for (substream_count = original_substreams;
	     substream_count > 1;
	     substream_count = (substream_count + merge_arity - 1)
		 / merge_arity) {

	    // Set up to process a given level.
//RAKESH
	    tp_assert (len == check_size,
		       "Current level stream not same length as input." <<
		       "\n\tlen = " << len <<
		       "\n\tcurrent_input->stream_len() = " <<
		       check_size << ".\n");

	    check_size = 0;

	    // Do we have enough main memory to merge all the
	    // substreams on the current level into the output stream?
	    // If so, then we will do so, if not then we need an
	    // additional level of iteration to process the substreams
	    // in groups.

	    if (substream_count <= merge_arity) {

//RAKESH   Open up the substream_count streams in which the
//         the runs input to the current merge level are packed
//         The names of these streams (storing the input runs)
//         can be constructed from  prefix_name[k % 2]

		for (ii = merge_arity - substream_count; ii < merge_arity;
		     ii++) {

#ifndef BTE_IMP_USER_DEFINED

		    stream_name_generator (working_disk,
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);

#endif

#ifdef  BTE_IMP_USER_DEFINED
		    stream_name_generator ("",
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);
#endif

		    current_input[ii] = new AMI_STREAM < T > (new_stream_name);
		    current_input[ii]->persist (PERSIST_DELETE);

		}

		// Merge them into the output stream.

		ae = AMI_single_merge (
		    (current_input + merge_arity -
		     substream_count), substream_count,
		    outstream, keyoffset, dummykey);

		if (ae != AMI_ERROR_NO_ERROR) {

		   TP_LOG_FATAL_ID ("AMI_ERROR " << 
				  ae << " returned by  AMI_single_merge()");
		    return ae;
		}
		// Delete the streams input to the above merge.

		for (ii = merge_arity - substream_count;
		     ii < merge_arity; ii++) {
		    if (current_input[ii]) {
			delete current_input[ii];

			current_input[ii] = NULL;
		    }

		}

		if (current_input) {
		    delete[]current_input;
		    current_input = NULL;
		}

		if (the_substreams) {
		    delete[]the_substreams;
		    the_substreams = NULL;
		}

	    } else {

		TP_LOG_DEBUG_ID ("Merging substreams to intermediate streams.");

		// Create the array of merge_arity stream pointers that
		// will each point to a stream containing runs output
		// at the current level k. 

		intermediate_tmp_stream = new AMI_STREAM<T>*[merge_arity];

//RAKESH   Open up the merge_arity streams in which the
//         the runs input to the current merge level are packed
//         The names of these streams (storing the input runs)
//         can be constructed from  prefix_name[k % 2]

		for (ii = 0; ii < merge_arity; ii++) {

//                      new_name_from_prefix(prefix_name[k % 2],(int) ii,
//                                            new_stream_name);

#ifndef BTE_IMP_USER_DEFINED
		    stream_name_generator (working_disk,
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
		    stream_name_generator ("",
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);
#endif

		    current_input[ii] = new AMI_STREAM < T > (new_stream_name);

		    current_input[ii]->persist (PERSIST_DELETE);

		}

		// Fool the OS into unmapping the current block of the
		// input stream so that blocks of the substreams can
		// be mapped in without overlapping it.  This is
		// needed for correct execution on HU-UX.
//RAKESH
//                current_input->seek(0);

		current_stream = merge_arity - 1;

		//For the first stream that we use to pack some    
		//of the output runs of the current merge level k.

//                new_name_from_prefix(prefix_name[(k+1) % 2],0,
//                                            new_stream_name);

#ifndef BTE_IMP_USER_DEFINED

		stream_name_generator (working_disk,
				       prefix_name[(k + 1) % 2],
				       current_stream, new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
		stream_name_generator ("",
				       prefix_name[(k + 1) % 2],
				       current_stream, new_stream_name);
#endif

		intermediate_tmp_stream[current_stream] = new
		    AMI_STREAM < T > (new_stream_name);

		intermediate_tmp_stream[current_stream]->persist
		    (PERSIST_PERSISTENT);

		int remaining_number_of_output_runs =
		    (substream_count + merge_arity - 1) / merge_arity;

		for (ii_streams = 0; ii_streams < (int) merge_arity;
		     ii_streams++) {
		    // If there are 12 runs to be distributed among 5 streams, 
		    // the first three get 2 and the last two  get 3 runs   

		    if (ii_streams <
			(int) (merge_arity -
			       (remaining_number_of_output_runs % merge_arity)))

			desired_runs_in_stream[ii_streams] =
			    remaining_number_of_output_runs / merge_arity;

		    else
			desired_runs_in_stream[ii_streams] =
			    (remaining_number_of_output_runs
			     + merge_arity - 1) / merge_arity;

		    Sub_Start(ii_streams) = 0;

		}

		runs_in_current_stream = 0;
		unsigned int merge_number = 0;

		// Loop through the substreams of the current stream,
		// merging as many as we can at a time until all are
		// done with.

		for (sub_start = 0, ii = 0, jj = 0; ii < substream_count; ii++) {

		    if (run_lengths(k % 2, merge_arity - 1 - jj, merge_number)
			!= 0) {

			sub_start = Sub_Start(merge_arity - 1 - jj);

			sub_end = sub_start +
			    run_lengths(k % 2, merge_arity - 1 -
					       jj, merge_number) - 1;

			Sub_Start(merge_arity - 1 - jj) +=
			    run_lengths(k % 2, merge_arity - 1 -
					       jj, merge_number);

			run_lengths(k % 2, merge_arity - 1 - jj, merge_number)
			    = 0;
		    } else {
			//This weirdness is caused by the way bte substream
			//constructor was designed.

			sub_end = Sub_Start(merge_arity - 1 - jj) - 1;
			sub_start = sub_end + 1;

			ii--;

		    }

		    //Open the new substream
		    current_input[merge_arity - 1 -
				  jj]->new_substream (AMI_READ_STREAM,
						      sub_start, sub_end,
						      (AMI_stream_base < T > **)
						      (the_substreams + jj));

		    // The substreams are read-once.
		    // If we've got all we can handle or we've seen
		    // them all, then merge them.

		    if ((jj >= merge_arity - 1) || (ii == substream_count - 1)) {

			tp_assert (jj <= merge_arity - 1,
				   "Index got too large.");

			//Check if the stream into which runs are cuurently 
			//being packed has got its share of runs. If yes,
			//delete that stream and construct a new stream 
			//appropriately.

			if (desired_runs_in_stream[current_stream] ==
			    runs_in_current_stream) {

			    //Make sure that the deleted stream persists on disk.
			    intermediate_tmp_stream[current_stream]->persist
				(PERSIST_PERSISTENT);

			    delete intermediate_tmp_stream[current_stream];

			    current_stream = (current_stream + merge_arity - 1)
				% merge_arity;

			    // Unless the current level is over, we've to generate 
			    //a new stream for the next set of runs.

			    if (remaining_number_of_output_runs > 0) {

//                        new_name_from_prefix(prefix_name[(k+1) % 2],
//                              current_stream, new_stream_name);

#ifndef BTE_IMP_USER_DEFINED
				stream_name_generator (working_disk,
						       prefix_name[(k + 1) % 2],
						       (int) current_stream,
						       new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
				stream_name_generator ("",
						       prefix_name[(k + 1) % 2],
						       (int) current_stream,
						       new_stream_name);
#endif

				intermediate_tmp_stream[current_stream] = new
				    AMI_STREAM < T > (new_stream_name);

				intermediate_tmp_stream[current_stream]->persist
				    (PERSIST_PERSISTENT);
				runs_in_current_stream = 0;
			    }
			}

			ae = AMI_single_merge (the_substreams,
					       jj + 1,
					       intermediate_tmp_stream
					       [current_stream], keyoffset,
					       dummykey);

			if (ae != AMI_ERROR_NO_ERROR) {
			    return ae;
			}

			for (ii_streams = 0; ii_streams < (int) jj + 1;
			     ii_streams++)
			    run_lengths((k + 1) % 2, current_stream,
				runs_in_current_stream) +=
				the_substreams[ii_streams]->stream_len ();

			merge_number++;

			//Decrement the counter corresp to number of runs 
			// still to be formed at current level

			remaining_number_of_output_runs--;

			// Delete input substreams. jj is currently the index
			// of the largest.

			for (ii_streams = 0; ii_streams < (int) jj + 1;
			     ii_streams++) {
			    if (the_substreams[ii_streams]) {
				delete the_substreams[ii_streams];

				the_substreams[ii_streams] = NULL;
			    }
			}

			jj = 0;

//RAKESH                The number of runs in the current_stream
//                      goes up by 1.

			runs_in_current_stream++;

		    } else {
			jj++;
		    }

		}

		if (intermediate_tmp_stream[current_stream]) {
		    delete intermediate_tmp_stream[current_stream];

		    intermediate_tmp_stream[current_stream] = NULL;
		}
		// Get rid of the current input streams and use the ones
		//output at the current level.
//RAKESH

		for (ii = 0; ii < merge_arity; ii++)
		    if (current_input[ii])
			delete current_input[ii];

		if (current_input) {
		    delete[]current_input;
		    current_input = NULL;
		}

		current_input = (AMI_STREAM < T > **)intermediate_tmp_stream;

	    }

	    k++;

	}

	//Monitoring prints.

	TP_LOG_DEBUG_ID ("Number of passes incl run formation is " << k +
		      1 );

	TP_LOG_DEBUG_ID ("AMI_partition_and_merge_Key: done");
	return AMI_ERROR_NO_ERROR;

    }

    assert (0);			// no return value - die - R..
}

//------------------------------------------------------------
template < class T, class KEY >
AMI_err AMI_replacement_selection_and_merge_Key (AMI_STREAM < T >
						 *instream,
						 AMI_STREAM < T >
						 *outstream,
						 int keyoffset,
						 KEY dummykey)
{
    AMI_err ae;
    TPIE_OS_OFFSET len;
    size_t sz_avail, sz_stream;
    size_t sz_substream;

    unsigned int ii, jj, kk;
    int ii_streams;

#ifndef BTE_IMP_USER_DEFINED
    char *working_disk;
#endif

    // Figure out how much memory we've got to work with.

    sz_avail = MM_manager.memory_available ();

#ifndef BTE_IMP_USER_DEFINED
    working_disk = tpie_tempnam ("AMI");
    //TP_LOG_DEBUG_ID(working_disk);
#endif

    // If the whole input can fit in main memory then just call
    // AMI_main_mem_merge() to deal with it by loading it once and
    // processing it.

    len = instream->stream_len ();
    instream->seek (0);

    if ((len * sizeof (T)) <= sz_avail) {

	if (len * (sizeof (T) * sizeof (qsort_item < KEY >)) > sz_avail)
	    // ie if you have dont have space for separate
	    // keysorting (good cache performance) followed by permuting 
	{

	    T *next_item;

	   TP_LOG_DEBUG_ID ("pre new");
	    T *mm_stream = new T[len];

	   TP_LOG_DEBUG_ID ("post new");

	    for (int i = 0; i < len; i++) {
		if ((ae = instream->read_item (&next_item)) !=
		    AMI_ERROR_NO_ERROR) return ae;
		mm_stream[i] = *next_item;
	    }

	    quick_sort_op ((T *) mm_stream, len);

	    for (int i = 0; i < len; i++) {
		if ((ae = outstream->write_item (mm_stream[i]))
		    != AMI_ERROR_NO_ERROR)
		    return ae;
	    }
	   TP_LOG_DEBUG_ID ("pre delete");
	    if (mm_stream) {
		delete[]mm_stream;
		mm_stream = NULL;
	    }
	   TP_LOG_DEBUG_ID ("post delete");
	} else {
	    //Use qsort on keys followed by permuting

	   TP_LOG_DEBUG_ID ("pre new");
	    T *mm_stream = new T[len];

	   TP_LOG_DEBUG_ID ("post new");
	    qsort_item < KEY > *qs_array = new qsort_item<KEY>[len];
	   TP_LOG_DEBUG_ID ("post new");
	    T *next_item;

	    for (int i = 0; i < len; i++) {
		if ((ae = instream->read_item (&next_item)) !=
		    AMI_ERROR_NO_ERROR) return ae;
		mm_stream[i] = *next_item;
		qs_array[i].keyval = *(KEY *) ((char *) next_item + keyoffset);
		qs_array[i].source = i;
	    }

	    quick_sort_op ((qsort_item < KEY > *)qs_array, len);

	    for (int i = 0; i < len; i++) {
		if (
		    (ae =
		     outstream->write_item (mm_stream[qs_array[i].source])) !=
		    AMI_ERROR_NO_ERROR) return ae;
	    }
	   TP_LOG_DEBUG_ID ("pre delete");
	    if (mm_stream) {
		delete[]mm_stream;
		mm_stream = NULL;
	    }
	   TP_LOG_DEBUG_ID ("post delete");
	    if (qs_array) {
		delete[]qs_array;
		qs_array = NULL;
	    }
	   TP_LOG_DEBUG_ID ("post delete");
	}

	return AMI_ERROR_NO_ERROR;

    } else {

	// The number of substreams that the original input stream
	// will be split into

	arity_t original_substreams;

	// The length, in terms of stream objects of type T, of the
	// original substreams of the input stream.  The last one may
	// be shorter than this.

	size_t sz_original_substream;

	// The initial temporary stream, to which substreams of the
	// original input stream are written.

	AMI_STREAM < T > **initial_tmp_stream;

	// The number of substreams that can be merged together at once.

	arity_t merge_arity;

	// A pointer to the buffer in main memory to read a memory load into.
	T *mm_stream;

	// Loop variables:

	// The stream being read at the current level.

	int runs_in_current_stream;

//RAKESH
	AMI_STREAM < T > **current_input;

	// The output stream for the current level if it is not outstream.

//RAKESH
	AMI_STREAM < T > **intermediate_tmp_stream;

	//TO DO
//RAKESH  (Hard coded prefixes) Ideally you be asking TPIE to give new names
	char *prefix_name[] = { "_0_", "_1_" };
	char itoa_str[5];

	// The size of substreams of *current_input that are being
	// merged.  The last one may be smaller.  This value should be
	// sz_original_substream * (merge_arity ** k) where k is the
	// number of iterations the loop has gone through.

	size_t current_substream_len;

	// The exponenent used to verify that current_substream_len is
	// correct.

	unsigned int k;

	TPIE_OS_OFFSET sub_start, sub_end;

	// How many substreams will there be?  The main memory
	// available to us is the total amount available, minus what
	// is needed for the input stream and the temporary stream.

	size_t mergeoutput_v;

	if ((ae = instream->main_memory_usage (&sz_stream,
					       MM_STREAM_USAGE_MAXIMUM)) !=
	    AMI_ERROR_NO_ERROR) {
	    return ae;
	}

	if ((ae = instream->main_memory_usage (&sz_substream,
					       MM_STREAM_USAGE_OVERHEAD)) !=
	    AMI_ERROR_NO_ERROR) {

	    return ae;
	}
	//Conservatively assume that the input and output streams
	//have not been accounted for in the bte_stream.

	sz_avail -= 2 * sz_stream;

	sz_original_substream = sz_avail - 2 * sz_stream;

//      Here the above var is in bytes: in AMI_partition_and_merge,
//      its in number of items of type T.

//RAKESH
// In our case merge_arity is determined differently than in the original
// implementation of AMI_partition_and_merge since we use several streams
// in each level.
// In our case net main memory required to carry out an R-way merge is
// (R+1)*MM_STREAM_USAGE_MAXIMUM  {R substreams for input runs, 1 stream for output}
// + R*MM_STREAM_USAGE_OVERHEAD   {One stream for each active input run: but while
//                                 the substreams use buffers, streams don't}
// + (R+1)*m_obj->space_usage_per_stream();
//
// The net memory usage for an R-way merge is thus
// R*(sz_stream + sz_substeam + m_obj->space_usage_per_stream()) + sz_stream +
// m_obj->space_usage_per_stream();
//

	//We can probably make do with a little less memory
	//if there is only a single binary merge pass required
	//but its too specialized a case to optimize for.

	if (sz_avail <= 3 * (sz_stream + sz_substream
			     + sizeof (merge_heap_element < KEY >))
	    //+ sz_stream + sizeof(merge_heap_element<KEY>)
	    ) {

	   TP_LOG_FATAL_ID
		("Insufficient Memory for AMI_replacement_selection_and_merge_Key()");
	    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
	}
	// Round the original substream length off to an integral
	// number of chunks.  This is for systems like HP-UX that
	// cannot map in overlapping regions.  It is also required for
	// BTE's that are capable of freeing chunks as they are
	// read.

	{
	    size_t sz_chunk_size = instream->chunk_size ();

	    //RAKESH: Why is this the ceiling instead of being the floor?

	    sz_original_substream = sz_chunk_size *
		((sz_original_substream + sz_chunk_size - 1) / sz_chunk_size);
	}

//The foll qty is a "to be determined" qty since the run lengths
// resulting from replacement selection are unknown.

	// Account for the space that a merge object will use.

	{
	    size_t sz_avail_during_merge =
		sz_avail - sz_stream - sz_substream - sz_stream -
		sizeof (merge_heap_element < KEY >);

	    size_t sz_stream_during_merge = sz_stream + sz_substream +
		sizeof (merge_heap_element < KEY >);

	    merge_arity = sz_avail_during_merge / sz_stream_during_merge;

	}

	// Make sure that the AMI is willing to provide us with the
	// number of substreams we want.  It may not be able to due to
	// operating system restrictions, such as on the number of
	// regions that can be mmap()ed in.

	{
	    int ami_available_streams = instream->available_streams ();

	    if (ami_available_streams != -1) {
		if (ami_available_streams <= 4) {
		    return AMI_ERROR_INSUFFICIENT_AVAILABLE_STREAMS;
		}

		if (merge_arity > (arity_t) ami_available_streams - 2) {
		    merge_arity = ami_available_streams - 2;
		   TP_LOG_DEBUG_ID
			("Reduced merge arity due to AMI restrictions.");

		}
	    }
	}

	TP_LOG_DEBUG_ID ("AMI_replacement_selection_and_merge(): merge arity = "
		      << merge_arity );

	if (merge_arity < 2) {
	   TP_LOG_FATAL_ID
		("Insufficient Memory for AMI_replacement_selection_and_merge_Key()");
	    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
	}
	// Create a temporary stream, then iterate through the
	// substreams, processing each one and writing it to the
	// corresponding substream of the temporary stream.
//RAKESH

	instream->seek (0);

	size_t check_size = 0;

	char computed_prefix[BTE_STREAM_PATH_NAME_LEN];
	char new_stream_name[BTE_STREAM_PATH_NAME_LEN];

	//Compute a prefix that will be sent to the run formation function,
	//since that is where the initial runs are formed.

#ifndef BTE_IMP_USER_DEFINED
	strcpy (computed_prefix, working_disk);
	//strcat(computed_prefix,"/");
	strcat (computed_prefix, prefix_name[0]);
#endif

#ifdef BTE_IMP_USER_DEFINED
	strcpy (computed_prefix, prefix_name[0]);
#endif

	//Conservatie estimate of the max possible number of runs during
	//run formation.
	int MaxRuns = instream->stream_len () /
	    (sz_original_substream /
	     (sizeof (run_formation_item < KEY >) + sizeof (T)));

	//Arrays to store the number of runs in each of the streams formed 
	//during each pass and the length of each of the runs.

	int RunsInStream[2][merge_arity],
	    RunLengths[2][merge_arity][(MaxRuns + merge_arity - 1) /
				       merge_arity];

	for (int i = 0; i < merge_arity; i++) {
	    RunsInStream[0][i] = 0;
	    RunsInStream[1][i] = 0;
	}

	KEY dummykey;		// This is only for the last argument to 

	//Run_Formation() that was added because of type 
	//unifcation problems.

	//Call the run formation function.

	if ((ae = Run_Formation_Algo_R_Key (instream,
					    merge_arity,
					    initial_tmp_stream,
					    computed_prefix,
					    sz_original_substream,
					    RunsInStream[0],
					    (int **) RunLengths[0],
					    (MaxRuns + merge_arity -
					     1) / merge_arity, keyoffset,
					    dummykey)) != AMI_ERROR_NO_ERROR) {
	   TP_LOG_FATAL_ID ("AMI Error " << 
			  ae << " in  Run_Formation_Algo_R_Key()");
	    return ae;

	}
	// Make sure the total length of the temporary stream is the
	// same as the total length of the original input stream.

	arity_t run_count = 0;

	for (int i = 0; i < merge_arity; i++) {
	    for (int j = 0; j < RunsInStream[0][i]; j++) {

		check_size += RunLengths[0][i][j];
	    }

	    run_count += RunsInStream[0][i];
	}

	if (check_size != instream->stream_len ()) {
	   TP_LOG_FATAL_ID
		("Run_Formation_Algo_R_Key() output different from input stream in length");
	    return AMI_ERROR_IO_ERROR;
	}

	tp_assert (instream->stream_len () == check_size,
		   "Stream lengths do not match:" <<
		   "\n\tinstream->stream_len() = " << instream->stream_len ()
		   << "\n\tinitial_tmp_stream->stream_len() = " << check_size
		   << ".\n");

	//We now delete the instream; note that it will be wiped off
	//disk if instream->persistence is set to PERSIST_DELETE
	//delete instream;

	// Set up the loop invariants for the first iteration of the
	// main loop.

	current_input = new AMI_STREAM<T> *[merge_arity];
	arity_t next_level_run_count;
	int run_start[merge_arity];

	// Pointers to the substreams that will be merged.

//RAKESH
	AMI_STREAM < T > **the_substreams = new AMI_STREAM<T>*[merge_arity];

	k = 0;

	// The main loop.  At the outermost level we are looping over
	// levels of the merge tree.  Typically this will be very
	// small, e.g. 1-3.

	while (run_count > 1) {

	    // Set up to process a given level.
//RAKESH

	    // Do we have enough main memory to merge all the
	    // substreams on the current level into the output stream?
	    // If so, then we will do so, if not then we need an
	    // additional level of iteration to process the substreams
	    // in groups.

	    if (run_count <= merge_arity) {

//RAKESH   Open up the run_count streams in which the
//         the runs input to the current merge level are packed
//         The names of these streams (storing the input runs)
//         can be constructed from  prefix_name[k % 2]

		for (ii = merge_arity - run_count; ii < merge_arity; ii++) {

#ifndef BTE_IMP_USER_DEFINED

		    stream_name_generator (working_disk,
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);

#endif

#ifdef  BTE_IMP_USER_DEFINED

		    stream_name_generator ("",
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);

#endif

		    current_input[ii] = new AMI_STREAM < T > (new_stream_name);
		    current_input[ii]->persist (PERSIST_DELETE);

		}

		// Merge them into the output stream.

		ae = AMI_single_merge (
		    (current_input + merge_arity -
		     run_count), run_count, outstream,
		    keyoffset, dummykey);

		if (ae != AMI_ERROR_NO_ERROR) {
		   TP_LOG_FATAL_ID ("AMI Error ");
		   TP_LOG_FATAL (ae);
		   TP_LOG_FATAL ("AMI_single_merge()");
		    return ae;
		}
		// Delete the substreams.
//RAKESH

		for (ii = merge_arity - run_count; ii < merge_arity; ii++) {

		    if (current_input[ii])
			delete current_input[ii];

		}

		// And the current input, which is an intermediate stream
		// of some kind.

		if (current_input) {
		    delete[]current_input;
		    current_input = NULL;
		}
		if (the_substreams) {
		    delete[]the_substreams;
		    the_substreams = NULL;
		}

		run_count = 1;

	    } else {

		TP_LOG_DEBUG_ID
		    ("Merging substreams to an intermediate stream.");

		// Create the array of merge_arity stream pointers that
		// will each point to a stream containing runs output
		// at the current level k.

		// Note that the array RunLengths[k % 2][ii] contains lengths of
		// the RunsInStream[k % 2][ii] runs in current_input stream
		// ii. 

		//Number of runs in the next level.
		next_level_run_count =
		    (run_count + merge_arity - 1) / merge_arity;

		intermediate_tmp_stream = new AMI_STREAM<T>*[merge_arity];

//RAKESH   Open up the merge_arity streams in which the
//         the runs input to the current merge level are packed
//         The names of these streams (storing the input runs)
//         can be constructed from  prefix_name[k % 2]

		for (ii = 0; ii < merge_arity; ii++) {

//                      new_name_from_prefix(prefix_name[k % 2],(int) ii,
//                                            new_stream_name);

#ifndef BTE_IMP_USER_DEFINED

		    stream_name_generator (working_disk,
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED

		    stream_name_generator ("",
					   prefix_name[k % 2],
					   (int) ii, new_stream_name);

#endif

		    //Construct the stream
		    current_input[ii] = new AMI_STREAM < T > (new_stream_name);

		    current_input[ii]->persist (PERSIST_DELETE);

		}

		//Stream counter
		int current_stream = merge_arity - 1;

		//Construct the first stream that we use to pack some
		//of the output runs of the current merge level k.

#ifndef BTE_IMP_USER_DEFINED

		stream_name_generator (working_disk,
				       prefix_name[(k + 1) % 2],
				       current_stream, new_stream_name);
#endif

#ifdef BTE_IMP_USER_DEFINED
		stream_name_generator ("",
				       prefix_name[(k + 1) % 2],
				       current_stream, new_stream_name);

#endif

		intermediate_tmp_stream[current_stream] =
		    new AMI_STREAM < T > (new_stream_name);

		intermediate_tmp_stream[current_stream]->persist
		    (PERSIST_PERSISTENT);

		//Number of output runs that remain to be generated at this level.
		int remaining_number_of_output_runs =
		    (run_count + merge_arity - 1) / merge_arity;

		//Determine the number of runs that will go in each of the streams
		//that will be output at this level.

		for (ii_streams = 0; ii_streams < merge_arity; ii_streams++) {

		    if (ii_streams <
			(merge_arity -
			 (remaining_number_of_output_runs % merge_arity)))

			RunsInStream[(k + 1) % 2][ii_streams] =
			    remaining_number_of_output_runs / merge_arity;

		    else
			RunsInStream[(k + 1) % 2][ii_streams] =
			    (remaining_number_of_output_runs + merge_arity - 1) /
			    merge_arity;
		    run_start[ii_streams] = 0;

		}

		runs_in_current_stream = 0;

		// Loop through the substreams of the current stream,
		// merging as many as we can at a time until all are
		// done with.

		mergeoutput_v = 0;
		int merge_number = 0;

		for (ii = 0, jj = 0; ii < run_count; ii++, jj++) {

		    //Runs can be of various lengths; so need to have 
		    //appropriate starting and ending points for substreams.

		    sub_start = run_start[merge_arity - 1 - jj];

		    sub_end = sub_start +
			RunLengths[k % 2][merge_arity - 1 - jj][merge_number] -
			1;

		    run_start[merge_arity - 1 - jj] +=
			RunLengths[k % 2][merge_arity - 1 - jj][merge_number];

		    //The weirdness below is because of the nature of the
		    // substream arguments.

		    if (sub_end >=
			current_input[merge_arity - 1 - jj]->stream_len ()) {

			sub_end =
			    current_input[merge_arity - 1 - jj]->stream_len () -
			    1;

			if (sub_start >
			    current_input[merge_arity - 1 - jj]->stream_len ())

			    sub_start = sub_end + 1;

		    }

		    mergeoutput_v += sub_end - sub_start + 1;

		    if (sub_end - sub_start + 1 == 0)
			ii--;

		    //NOTE:If the above condition is true it means that 
		    // the run just  encountered is a dummy run;
		    // the last merge of a pass  has
		    //   ( merge_arity - (run_count % merge_arity) )
		    // dummy runs; no other merge of the pass has any dummy run.

		    current_input[merge_arity - 1 -
				  jj]->new_substream (AMI_READ_STREAM,
						      sub_start, sub_end,
						      (AMI_stream_base < T > **)
						      (the_substreams + jj));

		    // If we've got all we can handle or we've seen
		    // them all, then merge them.

		    if ((jj >= merge_arity - 1) || (ii == run_count - 1)) {

			tp_assert (jj <= merge_arity - 1,
				   "Index got too large.");

			//Check to see if the current intermediate_tmp_stream
			//contains as many runs as it should; if yes, then
			//destroy (with PERSISTENCE) that stream and 
			//construct the next intermediate_tmp_stream. 

			if (RunsInStream[(k + 1) % 2][current_stream]
			    == runs_in_current_stream) {

			    intermediate_tmp_stream[current_stream]->persist
				(PERSIST_PERSISTENT);

			    delete intermediate_tmp_stream[current_stream];

			    current_stream =
				(current_stream + merge_arity - 1) % merge_arity;

			    // Unless the current level is over, we've to 
			    //generate a new stream for the next set of runs.

			    if (remaining_number_of_output_runs > 0) {

#ifndef BTE_IMP_USER_DEFINED

				stream_name_generator (working_disk,
						       prefix_name[(k + 1) % 2],
						       (int) current_stream,
						       new_stream_name);

#endif

#ifdef BTE_IMP_USER_DEFINED

				stream_name_generator ("",
						       prefix_name[(k + 1) % 2],
						       (int) current_stream,
						       new_stream_name);

#endif

				intermediate_tmp_stream[current_stream] =
				    new AMI_STREAM < T > (new_stream_name);

				intermediate_tmp_stream[current_stream]->persist
				    (PERSIST_PERSISTENT);
				runs_in_current_stream = 0;
			    }
			}
			// The merge should append to the output stream, since
			// AMI_single_merge() does not rewind the
			// output before merging.

			ae = AMI_single_merge (the_substreams,
					       jj + 1,
					       intermediate_tmp_stream
					       [current_stream], keyoffset,
					       dummykey);

			if (ae != AMI_ERROR_NO_ERROR) {
			   TP_LOG_FATAL_ID ("AMI Error ");
			   TP_LOG_FATAL (ae);
			   TP_LOG_FATAL ("AMI_single_merge()");
			    return ae;
			}

			RunLengths[(k + 1) % 2][current_stream]
			    [runs_in_current_stream] = mergeoutput_v;

			merge_number++;

//RAKESH Decrement the counter corresp to number of runs still to be
// formed at current level

			mergeoutput_v = 0;
			remaining_number_of_output_runs--;

			// Delete the substreams.  jj is currently the index
			// of the largest, so we want to bump it up before the
			// idiomatic loop.

			for (jj++; jj--;) {
			    if (the_substreams[jj]) {
				delete the_substreams[jj];

				the_substreams[jj] = NULL;
			    }
			}

			// Now jj should be -1 so that it gets bumped
			// back up to 0 before the next iteration of
			// the outer loop.
			tp_assert ((jj == -1), "Index not reduced to -1.");

//RAKESH                Advance the starting position within each of the
//                      current_input streams by the input run length
//                      of merge level k.

//RAKESH                The number of runs in the current_stream
//                      goes up by 1.

			runs_in_current_stream++;

		    }

		}

		if (intermediate_tmp_stream[current_stream]) {
		    delete intermediate_tmp_stream[current_stream];

		    intermediate_tmp_stream[current_stream] = NULL;
		}
		// Get rid of the current input stream and use the next one.

		for (ii = 0; ii < merge_arity; ii++) {
		    if (current_input[ii])
			delete current_input[ii];
		}

		if (current_input) {
		    delete[]current_input;
		    current_input = NULL;
		}

		current_input = (AMI_STREAM < T > **)intermediate_tmp_stream;

		run_count = next_level_run_count;
	    }

	    k++;

	}

	return AMI_ERROR_NO_ERROR;
    }
}

//------------------------------------------------------------
template < class T, class KEY >
AMI_err
Run_Formation_Algo_R_Key (AMI_STREAM < T > *instream,
			  arity_t arity,
			  AMI_STREAM < T > **outstreams,
			  char *computed_prefix,
			  size_t available_mem,
			  int *LRunsInStream,
			  int **LRunLengths,
			  int dim2_LRunLengths, int offset_to_key,
			  KEY dummykey)
{

    char local_copy[BTE_STREAM_PATH_NAME_LEN];

    strcpy (local_copy, computed_prefix);

    AMI_err ami_err;

//For now we are assuming that the key is of type int 
//and that the offset of the key within an item of type
//T is offset_to_key=0

//Define the proper structure for algorithm R of Vol 3

//What is called "P" in algorithm R in Vol 3. (Avg run length is 2P)
    unsigned int Number_P =
	available_mem / (sizeof (run_formation_item < KEY >) + sizeof (T));

    run_formation_item < KEY > *Array_X = new run_formation_item<KEY>[Number_P];
    T *Item_Array = new T[Number_P];

    T *ptr_to_record;
    unsigned int tempint;
    unsigned int Var_T;
    unsigned int curr_run_length = 0;

//We will  write first run into stream arity-1, then next run into
//stream arity-2, and so on in a round robin manner.
    int current_stream = arity - 1;

    char int_to_string[5], new_stream_name[BTE_STREAM_PATH_NAME_LEN];

    outstreams = new AMI_STREAM<T>*[arity];

    int *Cast_Var = (int *) LRunLengths;
    int MaxRuns =
	instream->stream_len () / (available_mem /
				   sizeof (run_formation_item < KEY >));

    int RF_Cntr = 0;

    short RMAX = 0;
    short RC = 0;
    KEY LASTKEY;			//Should be guaranteed to be initialized to something greater than 

    //the key value of the first item in instream, for correctness.

    int Q = 0;
    short RQ = 0;

    for (unsigned int j = 0; j < Number_P; j++) {
	Array_X[j].Loser = j;
	Array_X[j].RunNumber = 0;
	Array_X[j].ParentExt = (Number_P + j) / 2;
	Array_X[j].ParentInt = j / 2;
	Array_X[j].RecordPtr = j;
    }

 Step_R2:
    if (RQ != RC) {
	if (RC >= 1) {

	    Cast_Var[current_stream * dim2_LRunLengths
		     + LRunsInStream[current_stream]] = curr_run_length;

	    LRunsInStream[current_stream]++;

	    delete outstreams[current_stream];

	    current_stream = (current_stream + arity - 1) % arity;
	    RF_Cntr += curr_run_length;
	}

	if (RQ > RMAX)
	    goto Step_End;
	else
	    RC = RQ;

	// Now construct the possibly previously destroyed stream for
	// new run and seek to its end.

	//Compute the name for the stream
	sprintf (int_to_string, "%d", current_stream);
	strcpy (new_stream_name, local_copy);
	strcat (new_stream_name, int_to_string);

	// Use the appropriate constructor.
	//BEGIN CONSTRUCT STREAM

#ifdef BTE_IMP_USER_DEFINED

	outstreams[current_stream] = new AMI_STREAM < T > (new_stream_name);

#else				//! BTE_IMP_USER_DEFINED******************************************************

	outstreams[current_stream] = new AMI_STREAM < T > (new_stream_name);

#endif				//****************************************************************************

	//END CONSTRUCT STREAM

	outstreams[current_stream]->persist (PERSIST_PERSISTENT);

	outstreams[current_stream]->
	    seek (outstreams[current_stream]->stream_len ());

	// Now set length of currently being formed run to zero.

	curr_run_length = 0;

    }
// End of Step_R2

 Step_R3:
    if (RQ == 0) {

	if ((ami_err = instream->read_item (&ptr_to_record))
	    != AMI_ERROR_NO_ERROR) {
	    if (ami_err == AMI_ERROR_END_OF_STREAM) {
		RQ = RMAX + 1;
		goto Step_R5;
	    }
	    return ami_err;
	}
	//Copy the most recently read item into item array loc
	// Array_X[Q].RecordPtr            

	Item_Array[Array_X[Q].RecordPtr] = *ptr_to_record;
	Array_X[Q].Key = *(KEY *) ((char *) ptr_to_record + offset_to_key);

	//The above portion is actually carried out in Step R4 in Vol 3's 
	// description of Algorithm R. But here we carry it out in Step
	// R3 itself so that we can efficiently simulate LASTKEY=Infinity

	//We've made sure that we read the first record from instream
	// Now we set LASTKEY to be one more than that first record's key
	// so that it simulates LASTKEY=Infinity

	LASTKEY = *(KEY *) ((char *) ptr_to_record + offset_to_key);
	++LASTKEY;		//LASTKEY = LASTKEY+1;

    }

    else {

	if (
	    (ami_err =
	     outstreams[current_stream]->write_item (Item_Array
						     [Array_X[Q].
						      RecordPtr])) !=
	    AMI_ERROR_NO_ERROR) {
	    return ami_err;
	}

	LASTKEY = Array_X[Q].Key;

	curr_run_length++;

	// The foll portion is actually carried out in Step R4 in Vol 3's 
	// description of Algorithm R. But here we carry it out in Step
	// R3 itself so that we can efficiently simulate LASTKEY=Infinity

	if ((ami_err = instream->read_item (&ptr_to_record))
	    != AMI_ERROR_NO_ERROR) {
	    if (ami_err == AMI_ERROR_END_OF_STREAM) {
		RQ = RMAX + 1;
		goto Step_R5;
	    }
	    return ami_err;
	}

	Item_Array[Array_X[Q].RecordPtr] = *ptr_to_record;
	Array_X[Q].Key = *(KEY *) ((char *) ptr_to_record + offset_to_key);

    }

 Step_R4:			// Array_X[Q] already contains a new item from input stream.
    if (Array_X[Q].Key < LASTKEY) {

	// Array_X[Q].Record cannot go into the present run so :

	RQ = RQ + 1;
	if (RQ > RMAX)
	    RMAX = RQ;
    }

 Step_R5:

    Var_T = Array_X[Q].ParentExt;

 Step_R6:
    if (
	(Array_X[Var_T].RunNumber < RQ) ||
	((Array_X[Var_T].RunNumber == RQ) &&
	 // KEY(LOSER(T)) < KEY(Q)
	 Array_X[Array_X[Var_T].Loser].Key < Array_X[Q].Key)
	) {
	// Swap LOSER(T) and Q 
	tempint = Array_X[Var_T].Loser;
	Array_X[Var_T].Loser = Q;
	Q = tempint;

	//Swap RN(T) and RQ
	tempint = Array_X[Var_T].RunNumber;
	Array_X[Var_T].RunNumber = RQ;
	RQ = tempint;
    }

 Step_R7:
    if (Var_T == 1) {
	goto Step_R2;
    } else {
	Var_T = Array_X[Var_T].ParentInt;
	goto Step_R6;
    }

 Step_End:delete Array_X;
    delete Item_Array;

    delete[]outstreams;

    return AMI_ERROR_NO_ERROR;

}

#endif // _AMI_OPTIMIZED_MERGE_H
