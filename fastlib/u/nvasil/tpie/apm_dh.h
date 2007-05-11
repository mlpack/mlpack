// DK Summer 05 TODO:
// get rid of dh extension, convert to ami_merge
// perhaps grab merge calls from ami_sort_single_dh
// check mem usage thoroughly 
// get rid of varArrays where not needed
// change "substream" to "run" where appropriate
// get rid of User defined BTE junk
// Document better

// File: apm_dh.h 

#ifndef _APM_DH_H
#define _APM_DH_H
// **************************************************************************
// *                                                                        *
// *  This include file contains the routine                                *
// *    AMI_single_merge                                                    *
// *  used in several of TPIE's merge variants                              *
// *                                                                        *
// **************************************************************************
// 	$Id: apm_dh.h,v 1.18 2005/08/24 19:32:38 adanner Exp $	

// Get definitions for working with Unix and Windows
#include <portability.h>

// Includes needed from TPIE
#include <ami_stream.h>
#include <tpie_tempnam.h>
#include <mergeheap_dh.h>	   //For templated heaps
#include <quicksort.h>		//For templated qsort_items

typedef int arity_t;

// **************************************************************************
// *                                                                        *
// *   A M I _ s i n g l e _ m e r g e _ d h                                *
// *                                                                        *
// * This is a common merge routine for all of the AMI_merge, AMI_ptr_merge *
// * and AMI_key_merge entry points. It is also used by the sort entry      *
// * points AMI_sort, AMI_ptr_sort and AMI_key_sort and by the routine      *
// * AMI_partition_and_merge.  Differences are encapsulated within the      *
// * merge heap object 'MergeHeap'. It is assumed that MergeHeap.allocate   *
// * was called before entering AMI_single_merge_dh.                        *
// *                                                                        *
// **************************************************************************

template < class T, class M >
    AMI_err 
AMI_single_merge_dh (AMI_STREAM < T > **inStreams, arity_t arity,
		     AMI_STREAM < T > *outStream,  M MergeHeap,
         TPIE_OS_OFFSET cutoff=-1  )
{

   TPIE_OS_SIZE_T i;
   AMI_err ami_err;

   TPIE_OS_OFFSET* nread = new TPIE_OS_OFFSET[arity]; 
   
   //Pointers to current leading elements of streams
   T** in_objects = new T*[arity];

   // **************************************************************
   // * Read first element from stream. Do not rewind! We may read *
   // * more elements from the same stream later.                  *
   // **************************************************************

   for (i = 0; i < arity; i++) {

     if ((ami_err = inStreams[i]->read_item (&(in_objects[i]))) !=
         AMI_ERROR_NO_ERROR) {
       if (ami_err == AMI_ERROR_END_OF_STREAM) {
         in_objects[i] = NULL;
       } else {
         delete[] in_objects;
         return ami_err;
       }
     } else {
       MergeHeap.insert( in_objects[i], i );
     }
     nread[i]=1; 
   }

   // *********************************************************
   // * Build a heap from the smallest items of each stream   *
   // *********************************************************

   MergeHeap.initialize ( );

   // *********************************************************
   // * Perform the merge until the inputs are exhausted.     *
   // *********************************************************
   while (MergeHeap.sizeofheap() > 0) {

      i = MergeHeap.get_min_run_id ();
 
      if ((ami_err = outStream->write_item (*in_objects[i]))
          != AMI_ERROR_NO_ERROR) {
        delete[] in_objects;
        return ami_err;
      }

      //Check if we read as many elements as we are allowed to
      if( (cutoff != -1) && (nread[i]>=cutoff)){
        ami_err=AMI_ERROR_END_OF_STREAM;
      } 
      else {
        if ((ami_err = inStreams[i]->read_item (&(in_objects[i])))
            != AMI_ERROR_NO_ERROR) {
          if (ami_err != AMI_ERROR_END_OF_STREAM) {
            delete[] in_objects;
            return ami_err;
          }
        }
      } 
      if (ami_err == AMI_ERROR_END_OF_STREAM) {
        MergeHeap.delete_min_and_insert ((T *) NULL);
      } else { 
        nread[i]++;
        MergeHeap.delete_min_and_insert (in_objects[i]);
      }
   }//while

   //cleanup
   delete [] in_objects;
   delete [] nread;
   return AMI_ERROR_NO_ERROR;
}

/*******************************************************************
 *                                                                 *
 *           The actual AMI_merge calls                            *
 *                                                                 *
 *  These are the AMI_merge entry points for merging without the   *
 *  'merge_management_object' used by AMI_generalized_merge.       *
 *  These routines perform the special case of merging when the    *
 *  the required output is the original records interleaved        *
 *  according to a comparison operator or function.                *
 *******************************************************************/

/*
 Merging with a heap that contains the records to be merged: CMPR is
 the class of the comparison object, and must contain the method
 'compare' which is called from within the merge.

 TODO:
       1) check that memory management is done right
       2) add comparison operator and comparison function versions
       3) watch out for conflicts with AMI_merge versions in
*/

template < class T, class CMPR >
AMI_err AMI_merge (AMI_STREAM < T > **inStreams, arity_t arity,
    AMI_STREAM < T > *outStream, CMPR *cmp)
{
  // make a merge heap which uses the user's comparison object
  // and initialize it
  merge_heap_dh_obj<T,CMPR> mrgheap (cmp);
  mrgheap.allocate (arity);
  //Rewind all the input streams
  for(int i=0; i<arity; i++){ inStreams[i]->seek(0); }
  return AMI_single_merge_dh ( inStreams, arity, outStream, mrgheap);
}

// Merging with a heap that keeps a pointer to the records rather than
// the records themselves: CMPR is the class of the comparison object,
// and must contain the method 'compare' which is called from within
// the merge.

// TODO:
//       1) check that memory management is done right

template < class T, class CMPR >
AMI_err AMI_ptr_merge (AMI_STREAM < T > **inStreams, arity_t arity,
    AMI_STREAM < T > *outStream, CMPR *cmp)
{
  // make a merge heap of pointers which uses the user's comparison
  // object and initialize it
  merge_heap_pdh_obj<T,CMPR> mrgheap (cmp);
  mrgheap.allocate (arity);
  //Rewind all the input streams
  for(int i=0; i<arity; i++){ inStreams[i]->seek(0); }

  return AMI_single_merge_dh ( inStreams, arity, outStream, mrgheap);
}

// Merging with a heap that contains copies of the keys from the
// records being merged, rather than the records themselves:

// The comparison object "cmp", of (user-defined) class represented by
// CMPR, must have a member function called "compare" which is used
// for merging the input streams, and a member function called "copy"
// which is used for copying the key (of type KEY) from a record of
// type T (the type to be sorted).

// TODO:
//       1) check that memory management is done right

template<class T, class KEY, class CMPR>
AMI_err AMI_key_merge (AMI_STREAM < T > **inStreams, arity_t arity,
		AMI_STREAM < T > *outStream, CMPR *cmp)
{
	// make a key merge heap which uses the user's comparison object
	// and initialize it
	merge_heap_dh_kobj<T,KEY,CMPR> mrgheap (cmp);
	mrgheap.allocate (arity);
  //Rewind all the input streams
  for(int i=0; i<arity; i++){ inStreams[i]->seek(0); }

	return AMI_single_merge_dh ( inStreams, arity, outStream, mrgheap);
}

#endif //_APM_DH_H
