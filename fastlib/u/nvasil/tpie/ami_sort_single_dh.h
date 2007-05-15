//
// File: ami_sort_single_dh.h
//
// $Id: ami_sort_single_dh.h,v 1.18 2005/08/24 19:32:38 adanner Exp $
//
// This file contains the templated routines
//     1) AMI_sort:
//          a) AMI_err AMI_sort(AMI_STREAM<T> *instream, 
//                              AMI_STREAM<T> *outstream)
//          b) AMI_err AMI_sort(AMI_STREAM<T> *instream, 
//                              AMI_STREAM<T> *outstream,
//                              CMPR *cmp)
//     2) AMI_ptr_sort
//          a) AMI_err AMI_ptr_sort(AMI_STREAM<T> *instream, 
//                                  AMI_STREAM<T> *outstream)
//          b) AMI_err AMI_ptr_sort(AMI_STREAM<T> *instream, 
//                                  AMI_STREAM<T> *outstream,
//                                  CMPR *cmp)
//     3) AMI_key_sort
//          a) AMI_err AMI_key_sort(AMI_STREAM<T> *instream, 
//                                  AMI_STREAM<T> *outstream,
//                                  KEY dummykey, 
//                                  CMPR *cmp) 
//
// and the supporting class
//     sort_manager : a base class for sort managers
//     to actually sort, given an internal sort implementation and 
//     a single merge implementation
// 

// The routines AMI_partition_and_merge and AMI_single_merge are also
// used, and can be found in file apm_dh.h.  Besides the sort manager
// class, the sort routines are parameterized by an appropriate "merge
// heap class" and instantiate an object of that type. These classes
// can be found in file mergeheap_dh.h.

 
#ifndef _AMI_SORT_SINGLE_DH_H
#define _AMI_SORT_SINGLE_DH_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"
#include "u/nvasil/tpie/ami_stream.h" 
#include "u/nvasil/tpie/tpie_tempnam.h"
#include "u/nvasil/tpie/mergeheap_dh.h"  //For templated heaps
#include "u/nvasil/tpie/internal_sort.h" // Contains classes for sorting internal runs
                           // using different comparison types
#include <math.h> //for log, ceil, etc.

#ifndef AMI_STREAM_IMP_SINGLE
#warning Including __FILE__ when AMI_STREAM_IMP_SINGLE undefined.
#endif

typedef int arity_t;

// A class of merge objects for merge sorting objects of type T.  We
// will actually use one of two subclasses of this class which use
// either a comparison object,  or the binary comparison operator <.

template <class T, class I, class M>
class sort_manager{
 private:
    AMI_STREAM<T>* inStream;   
    AMI_STREAM<T>* outStream;   
    AMI_err         ae;           //For catching error codes
    TPIE_OS_OFFSET  nInputItems;  //Number of items in inStream;
    TPIE_OS_OFFSET  mmBytesAvail; //Amount of spare memory we can use
    TPIE_OS_SIZE_T  mmBytesPerStream; //Memory consumed by each Stream obj
 
    bool bProgress; //flag indicating if we show progress bar
    TPIE_OS_OFFSET progCount;

    bool use2xSpace; //flag to indicate if we are doing a 2x sort

    // The maximum number of stream items of type T that we can
    // sort in internal memory
    TPIE_OS_OFFSET nItemsPerRun;

    TPIE_OS_OFFSET nRuns; //The number of sorted runs left to merge
    arity_t mrgArity; //Max runs we can merge at one time

    // The output stream to which we are currently writing runs
    AMI_STREAM<T>* curOutputRunStream;

    // The mininum number of runs in each output stream
    // some streams can have one additional run
    TPIE_OS_OFFSET minRunsPerStream; 
    // The number of extra runs or the number of streams that
    // get one additional run.
    arity_t nXtraRuns;

    // The last run can have fewer than nItemsPerRun;
    TPIE_OS_OFFSET nItemsInLastRun;
    // How many items we will sort in a given run
    TPIE_OS_OFFSET nItemsInThisRun;
    // For each output stream, how many runs it should get
    TPIE_OS_OFFSET runsInStream;

    // A suffix to use in forming output file names. During the merge phase
    // we keep two sets of files, the input files and the output files to
    // which we are merging. The input file suffix is the opposite of the
    // output file suffix. After merging one level, the output streams
    // become the input for the next level.
    char *suffixName[2];
    // A buffer for building the output file names
    char   newName [BTE_STREAM_PATH_NAME_LEN];
    //prefix of temp files created during sort
    char *working_disk;

    AMI_err start_sort(); //high level wrapper to full sort 
    AMI_err compute_sort_params(); //compute nInputItems, mrgArity, nRuns
    AMI_err partition_and_sort_runs(); //make initial sorted runs
    AMI_err merge_to_output(); //loop over merge tree, create output stream
    // Merge a single group mrgArity streams to an output stream
    AMI_err single_merge(AMI_STREAM<T>**, arity_t, AMI_STREAM<T>*,
        TPIE_OS_OFFSET);

    //helper function for creating filename
    inline void make_name(char *prepre, char *pre, int id, char *dest);

 public:
    sort_manager(I isort, M mheap);
    ~sort_manager(){};
    I InternalSorter;
    M MergeHeap;
    //A version that uses 3x space and saves input stream
    AMI_err sort(AMI_STREAM<T>* in, AMI_STREAM<T>* out, bool progress);
    //A version that uses 2x space and overwrites input stream
    AMI_err sort(AMI_STREAM<T>* in, bool progress); 
};

template <class T, class I, class M>
sort_manager<T, I, M>::sort_manager(I isort, M mheap):
  InternalSorter(isort), MergeHeap(mheap)
{
 suffixName[0]="_0_";
 suffixName[1]="_1_";
 //prefix of temp files created during sort
 working_disk = tpie_tempnam("AMI");
};

template<class T, class I, class M>
AMI_err sort_manager<T,I,M>::sort(AMI_STREAM<T>* in, AMI_STREAM<T>* out,
                                  bool progress=false){

  //This version saves the original input and uses 3x space
  //(input, current temp runs, output runs)
  
  bProgress=progress;
  inStream=in;
  outStream=out;
  use2xSpace=false;
  // Basic checks that input is ok
  if(inStream==NULL || outStream==NULL) { return AMI_ERROR_NULL_POINTER;}
  if(!inStream || !outStream) { return AMI_ERROR_OBJECT_INVALID; }
  if(inStream->stream_len() < 2) { return AMI_SORT_ALREADY_SORTED; }
  
  // Else, there is something to sort, do it
  return start_sort();
}

template<class T, class I, class M>
AMI_err sort_manager<T,I,M>::sort(AMI_STREAM<T>* in, bool progress=false){

  //This version overwrites the original input and uses 2x space
  //The input stream is truncated to length 0 after forming initial runs
  //and only two levels of the merge tree are on disk at any one time.
  bProgress=progress;
  inStream=in;
  outStream=in; //output destination is same as input
  use2xSpace=true;
  // Basic checks that input is ok
  if(inStream==NULL) { return AMI_ERROR_NULL_POINTER;}
  if(!inStream) { return AMI_ERROR_OBJECT_INVALID; }
  if(inStream->stream_len() < 2) { return AMI_SORT_ALREADY_SORTED; }
  
  // Else, there is something to sort, do it
  return start_sort();
}

template<class T, class I, class M>
AMI_err sort_manager<T,I,M>::start_sort(){

  TP_LOG_DEBUG_ID ("sort_manager::sort START");
  if(bProgress){ cout << "\n----Starting TPIE Sort----" << endl; }
  // ********************************************************************
  // * PHASE 1: See if we can sort the entire stream in internal memory *
  // * without the need to use general merge sort                       *
  // ********************************************************************


  // Figure out how much memory we've got to work with.
  mmBytesAvail = MM_manager.memory_available();

  // Space for internal buffers for the input and output stream may not
  // have been allocated yet. Query the space usage and subtract.
  if ((ae = inStream->main_memory_usage
        (&mmBytesPerStream,MM_STREAM_USAGE_MAXIMUM))
      != AMI_ERROR_NO_ERROR) {
    TP_LOG_DEBUG_ID ("Error returned from main_memory_usage");
    return ae;
  }

  TP_LOG_DEBUG_ID ("BTE says we use at most "<< mmBytesPerStream
      << "bytes per stream");

  // This is how much we can use for internal sort if
  // we are not doing general merge sort
  mmBytesAvail -= 2 * mmBytesPerStream;

  // Check if all input items can be sorted internally using less than
  // mmBytesAvail
  nInputItems = inStream->stream_len();
  inStream->seek (0);
  if (nInputItems<InternalSorter.MaxItemCount(mmBytesAvail)){
    // allocate the internal array items
    InternalSorter.allocate(nInputItems);
    // load the items into main memory, sort, and write to output.
    // InternalSorter also checks if inStream/outStream are the same and
    // truncates/rewrites inStream if they are. This probably should not
    // be the job of InternalSorter. TODO: build a cleaner interface
    if ((ae = InternalSorter.sort(inStream, outStream, nInputItems))
        != AMI_ERROR_NO_ERROR) {
      TP_LOG_FATAL_ID ("main_mem_operate failed");
      return ae;
    }
    // de-allocate the internal array of items
    InternalSorter.deallocate();
    if(bProgress){ cout << "----Finished TPIE Sort----\n" << endl; }
    return AMI_ERROR_NO_ERROR;
  }

  // ******************************************************************
  // * Input stream too large for main memory, use general merge sort *
  // ******************************************************************

  // PHASE 2: compute nItemsPerRun, nItemsPerRun, nRuns
  ae=compute_sort_params();
  if(ae != AMI_ERROR_NO_ERROR){ return ae; }

  // ********************************************************************
  // * By this point we have checked that we have valid input, checked  *
  // * that we indeed need an external memory sort, verified that we    *
  // * have enough memory to partition and at least do a binary merge.  *
  // * Also checked that we have enough file descriptors to  merge,     *
  // * and calculated the mrgArity and nItemsPerRun given memory        *
  // * constraints. We have also calculated nRuns for the initial       *
  // * number of runs we will partition into. Let's sort!               *
  // ********************************************************************
  
  // ********************************************************************
  // * WARNING: Since we accounted for all known memory usage in PHASE 2*
  // * be very wary of memory allocation via "new" or constructors from *
  // * this point on and make sure it was accounted for in PHASE 2      *
  // ********************************************************************

  // PHASE 3: partition and form sorted runs
  TP_LOG_DEBUG_ID ("Beginning general merge sort.");
  ae=partition_and_sort_runs();
  if(ae != AMI_ERROR_NO_ERROR){ return ae; }

  // PHASE 4: merge sorted runs to a single output stream
  ae=merge_to_output();
  if(ae != AMI_ERROR_NO_ERROR){ return ae; }

  // That's it
  if(bProgress){ cout << "----Finished TPIE Sort----\n" << endl; }
  return AMI_ERROR_NO_ERROR;
}

template<class T, class I, class M>
AMI_err sort_manager<T,I,M>::compute_sort_params(void){
  // ********************************************************************
  // * PHASE 2: Compute/check limits                                    *
  // * Compute the maximum number of items we can sort in main memory   *
  // * and the maximium number of sorted runs we can merge at one time  *
  // * Before doing any sorting, check that we can fit at least one item*
  // * in internal memory for sorting and that we can merge at least two*
  // * runs at at time                                                  *
  // *                                                                  *
  // * Memory needed for the run formation phase:                       *
  // * 2*mmBytesPerStream +                  {for input/output streams} *
  // * nItemsPerRun*space_per_sort_item() +  {for each item sorted }    *
  // * space_overhead_sort()                 {constant overhead in      *
  // *                                        sort management object    *
  // *                                        during sorting       }    *
  // *                                                                  *
  // * Memory needed for a D-way merge:                                 *
  // *  Cost per merge stream:                                          *
  // *   mmBytesPerStream+              {a open stream to read from}    *
  // *   space_per_merge_item()+        {used in internal merge heap}   *
  // *   sizeof(T*)+sizeof(off_t)       {arrays in single_merge()}      *
  // *   sizeof(AMI_STREAM<T>*)         {array element that points to   *
  // *                                    merge stream}                 *
  // *  Fixed costs:                                                    *
  // *    2*mmBytesPerStream+        {original input stream + output    *
  // *                                 of current merge}                *
  // *    space_overhead_merge()+    {fixed dynamic memory costs of     *
  // *                                 merge heap}                      *
  // *    3*space_overhead()         {overhead per "new" memory request *
  // *                                for allocating 2 arrays of streams*
  // *                                and two arrays in single_merge}   *
  // *                                                                  *
  // *  Total cost for D-way Merge:                                     *
  // *    D*(Cost per merge stream)+(Fixed costs)                       *
  // *                                                                  *
  // *  Any additional memory requests that call "new" directly or      *
  // *  indirectly should be documented and accounted for in this phase *
  // ********************************************************************

  TP_LOG_DEBUG_ID ("Computing merge sort parameters.");

  TPIE_OS_OFFSET mmBytesAvailSort; // Bytes available for sorting

  TP_LOG_DEBUG_ID ("Each object of size " << sizeof(T) << " uses "
      << InternalSorter.space_per_item () << " bytes "
      << "for sorting in memory");

  //Subtract off size of temp output stream
  //The size of the input stream was already subtracted from
  //mmBytesAvail
  mmBytesAvailSort=mmBytesAvail - mmBytesPerStream;

  nItemsPerRun=InternalSorter.MaxItemCount(mmBytesAvailSort);

  if(nItemsPerRun<1){
    TP_LOG_FATAL_ID ("Insufficient Memory for forming sorted runs");
    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
  }

  // Now we know the max number of Items we can sort in a single
  // internal memory run. Next, compute the number of runs we can
  // merge together at one time

  TPIE_OS_SIZE_T mmBytesPerMergeItem = mmBytesPerStream +
    MergeHeap.space_per_item() + sizeof(T*) +
    sizeof(TPIE_OS_OFFSET)+sizeof(AMI_STREAM<T>*);

  // Fixed cost of mergheap impl. + MM_manager overhead of allocating
  // an array of AMI_STREAM<T> ptrs (pending)
  // cost of Input stream already accounted for in mmBytesAvail..
  TPIE_OS_SIZE_T mmBytesFixedForMerge = MergeHeap.space_overhead() +
    mmBytesPerStream + 3*MM_manager.space_overhead();

  TPIE_OS_OFFSET mmBytesAvailMerge = mmBytesAvail - mmBytesFixedForMerge;
  // Need to support at least binary merge
  if(mmBytesAvailMerge<2*mmBytesPerMergeItem){
    TP_LOG_FATAL_ID ("Merge arity < 2 -- Insufficient memory for a merge.");
    return AMI_ERROR_INSUFFICIENT_MAIN_MEMORY;
  }

  // Cast down from TPIE_OS_OFFSET (type of mmBytesAvail).
  // mmBytesPerMergeItem is at least 1KB, so we are OK unless we
  // have more than 2 TerraBytes of memory. I look forward to the day
  // this comment seems silly and wrong
  mrgArity =
    (arity_t)(mmBytesAvail-mmBytesFixedForMerge)/mmBytesPerMergeItem;
  TP_LOG_DEBUG_ID("mem avail=" << mmBytesAvail-mmBytesFixedForMerge
      << " bytes per merge item=" <<  mmBytesPerMergeItem
      << " initial mrgArity=" << mrgArity);

  // Make sure that the AMI is willing to provide us with the
  // number of substreams we want.  It may not be able to due to
  // operating system restrictions, such as on the number of regions
  // that can be mmap()ed in, max number of file descriptors, etc.
  int availableStreams = inStream->available_streams ();

  // Merging requires an available streams/file decriptor for
  // each of the mrgArity input. We need one additional file descriptor
  // for the output of the current merge, so binary merge requires
  // three available streams.
  if (availableStreams < 3) {
    TP_LOG_FATAL_ID ("Not enough stream descriptors available " <<
        "to perform merge.");
    return AMI_ERROR_INSUFFICIENT_AVAILABLE_STREAMS;
  }

  // Can at least do binary merge. See if availableStreams limits
  // maximum mrgArity
  if (mrgArity > availableStreams - 1) {
    mrgArity = (availableStreams - 1);
    TP_LOG_WARNING_ID ("Reduced merge arity due to AMI restrictions.");
  }

  // The number of memory-sized runs that the original input stream
  // will be partitioned into.
  nRuns = ((nInputItems + nItemsPerRun - 1) /
      nItemsPerRun);

#ifdef TPIE_SORT_SMALL_MRGARITY
  // KEEP OUT!!!
  // This should not be done by the typical user and is only for
  // testing/debugging purposes. ONLY define this flag and set a value
  // if you know what you are doing.
  TP_LOG_WARNING_ID("Reducing merge arity due to compiler specified flag");
  if(mrgArity > TPIE_SORT_SMALL_MRGARITY) {
    mrgArity=TPIE_SORT_SMALL_MRGARITY;
  }
#endif // TPIE_SORT_SMALL_MRGARITY

#ifdef TPIE_SORT_SMALL_RUNSIZE
  // KEEP OUT!!!
  // This should not be done by the typical user and is only for
  // testing/debugging purposes ONLY define this flag and set a value
  // if you know what you are doing.
  TP_LOG_WARNING_ID("Reducing run size due to compiler specified flag");
  if(nItemsPerRun > TPIE_SORT_SMALL_RUNSIZE) {
    nItemsPerRun=TPIE_SORT_SMALL_RUNSIZE;
  }

  // need to adjust nRuns
  nRuns = ((nInputItems + nItemsPerRun - 1) / nItemsPerRun);
#endif // TPIE_SORT_SMALL_RUNSIZE

  //#define MINIMIZE_INITIAL_RUN_LENGTH
#ifdef MINIMIZE_INITIAL_RUN_LENGTH
  // If compiled with the above flag, try to reduce the length of
  // the initial sorted runs without increasing the merge tree height
  // This could be a speed-up if it is faster to quicksort many small
  // runs
  // and merge many small runs than it is to quicksort fewer long runs
  // and
  // merge them.
  TP_LOG_DEBUG_ID ("Minimizing initial run lengths without increasing" <<
      "the height of the merge tree.");

  // The tree height is the ceiling of the log base mrgArity of the
  // number of original runs.
  double tree_height = log((double)nRuns) / log((double)mrgArity);
  tp_assert (tree_height > 0, "Negative or zero tree height!");
  tree_height = ceil (tree_height);

  // See how many runs we could possibly fit in the tree without
  // increasing the height.
  double maxOrigRuns = pow ((double) mrgArity, tree_height);
  tp_assert (maxOrigRuns >= nRuns,
      "Number of permitted runs was reduced.");

  // How big will such runs be?
  double new_nItemsPerRun = ceil (nInputItems/ maxOrigRuns);
  tp_assert (new_nItemsPerRun <= nItemsPerRun,
      "Size of original runs increased.");

  // Update the number of items per run and the number of original runs
  nItemsPerRun = (TPIE_OS_SIZE_T) new_nItemsPerRun;

  TP_LOG_DEBUG_ID ("With long internal memory runs, nRuns = "
      << nRuns << '\n');

  nRuns = (nInputItems + nItemsPerRun - 1) / nItemsPerRun;

  TP_LOG_DEBUG_ID ("With shorter internal memory runs "
      << "and the same merge tree height, nRuns = "
      << nRuns << '\n');

  tp_assert (maxOrigRuns >= nRuns,
      "We increased the merge height when we weren't supposed to do so.");
#endif  // MINIMIZE_INITIAL_SUBSTREAM_LENGTH


  // If we have just a few runs, we don't need the
  // full mrgArity. This is the last change to mrgArity
  if(mrgArity>nRuns){mrgArity=nRuns;}

  // We should always end up with at least two runs
  // otherwise why are we doing it externally?
  tp_assert (nRuns > 1, "Less than two runs to merge!");
  // Check that numbers are consistent with input size
  tp_assert (nRuns * nItemsPerRun - nInputItems < nItemsPerRun,
      "Total expected output size is too large.");
  tp_assert (nInputItems - (nRuns - 1) * nItemsPerRun <= nItemsPerRun,
      "Total expected output size is too small.");

  if(bProgress){
    cout << "Input stream has " << nInputItems << " Items\n"
         << "Forming " << nRuns << " initial runs of at most "
         << nItemsPerRun << " items each\n" 
         << "Merge arity is " << mrgArity << endl;
  }
  
  TP_LOG_DEBUG_ID ("Input stream has " << nInputItems << " Items");
  TP_LOG_DEBUG_ID ("Max number of items per runs " << nItemsPerRun );
  TP_LOG_DEBUG_ID ("Initial number of runs " << nRuns );
  TP_LOG_DEBUG_ID ("Merge arity is " << mrgArity );

  return AMI_ERROR_NO_ERROR;
}
template<class T, class I, class M>
AMI_err sort_manager<T,I,M>::partition_and_sort_runs(void){
  // ********************************************************************
  // * PHASE 3: Partition                                               *
  // * Partition the input stream into nRuns of at most nItemsPerRun    *
  // * and sort them, and write them to temporay output files.          *
  // * The last run may have fewer than nItemsPerRun. To keep the number*
  // * of files down and to support sequential I/O, we distribute the   *
  // * nRuns evenly across mrgArity files, thus each file on disk holds *
  // * multiple sorted runs.                                            *
  // ********************************************************************

  // The mininum number of runs in each output stream
  // some streams can have one additional run
  minRunsPerStream = nRuns/mrgArity;
  // The number of extra runs or the number of streams that
  // get one additional run. This is less than mrgArity and
  // it is OK to downcast to an arity_t.
  nXtraRuns = (arity_t) (nRuns - minRunsPerStream*mrgArity);
  tp_assert(nXtraRuns<mrgArity, "Too many extra runs");

  // The last run can have fewer than nItemsPerRun;
  // general case
  nItemsInLastRun=(nInputItems % nItemsPerRun);
  if(nItemsInLastRun==0){
    // Input size is an exact multiple of nItemsPerStream
    nItemsInLastRun=nItemsPerRun;
  }

  // Initialize memory for the internal memory runs
  // accounted for in phase 2:  (nItemsPerRun*size_of_sort_item) +
  // space_overhead_sort
  InternalSorter.allocate(nItemsPerRun);

  TP_LOG_DEBUG_ID ("Partitioning and forming sorted runs.");

  // nItemsPerRun except for last run.
  nItemsInThisRun=nItemsPerRun;
  arity_t ii,jj;  //Some index vars

  // Rewind the input stream, we are about to begin
  inStream->seek(0);

  // ********************************************************************
  // * Partition and make initial sorted runs                           *
  // ********************************************************************
  TPIE_OS_OFFSET check_size = 0; //for debugging
  progCount=0; //for progress indication
  for( ii=0; ii<mrgArity; ii++){   //For each output stream
    // Make the output file name
    make_name(working_disk, suffixName[0], ii, newName);
    // Dynamically allocate the stream
    // We account for these mmBytesPerStream in phase 2 (output stream)
    curOutputRunStream = new AMI_STREAM<T>(newName);
    // How many runs should this stream get?
    // extra runs go in the LAST nXtraRuns streams so that
    // the one short run is always in the LAST output stream
    runsInStream = minRunsPerStream + ((ii >= mrgArity-nXtraRuns)?1:0);
    for( jj=0; jj < runsInStream; jj++ ) { // For each run in this stream
      // See if this is the last run
      if( (ii==mrgArity-1) && (jj==runsInStream-1)) {
        nItemsInThisRun=nItemsInLastRun;
      }
      // Sort it
      if(bProgress){
        progCount++;
        cout << "\rForming sorted run " << progCount << " of " << nRuns
             << " [" << setw(6) << setiosflags(ios::fixed) 
             << setprecision(2) 
             << ((1.*progCount)/nRuns)*100. << "%]" << flush;
      }
      if ((ae = InternalSorter.sort(inStream, curOutputRunStream, 
              nItemsInThisRun))!= AMI_ERROR_NO_ERROR)
      {
        TP_LOG_FATAL_ID ("main_mem_operate failed");
        return ae;
      }
    } // For each run in this stream
    // All runs created for this stream, clean up
    TP_LOG_DEBUG_ID ("Wrote " << runsInStream << " runs and "
        << curOutputRunStream->stream_len() << " items to file " << ii);
    check_size+=curOutputRunStream->stream_len();
    curOutputRunStream->persist(PERSIST_PERSISTENT);
    delete curOutputRunStream;
  }//For each output stream

  tp_assert(check_size == nInputItems, "item count mismatch");

  // Done with partitioning and initial run formation
  // free space associated with internal memory sorting
  InternalSorter.deallocate();
  if(bProgress){ cout << endl; } //newline
  if(use2xSpace){ 
    //recall outStream/inStream point to same file in this case
    inStream->truncate(0); //free up disk space
    inStream->seek(0);
  } 
  return AMI_ERROR_NO_ERROR;
}

template<class T, class I, class M>
AMI_err sort_manager<T,I,M>::merge_to_output(void){

  // ********************************************************************
  // * PHASE 4: Merge                                                   *
  // * Loop over all levels of the merge tree, reading mrgArity runs    *
  // * at a time from the streams at the current level and distributing *
  // * merged runs over mrgArity output streams one level up, until     *
  // * a single output stream exists                                    *
  // ********************************************************************

  // The input streams we from which will read sorted runs
  AMI_STREAM<T> **mergeInputStreams = new AMI_STREAM<T>*[mrgArity];

  //This mesage does not count space overhead per "new"
  //should it?
  TP_LOG_DEBUG_ID("Allocated " << sizeof(AMI_STREAM<T>*)*mrgArity
      << " bytes for " << mrgArity << " merge input stream pointers."
      << " Mem. avail. is " << MM_manager.memory_available () );

  // the number of iterations the main loop has gone through,
  // the height of the merge tree log_{M/B}(N/B), typically 1 or 2
  int mrgHeight = 0;
  int treeHeight; //for progress
  TPIE_OS_OFFSET ii,jj; //index vars

  MergeHeap.allocate( mrgArity ); //Allocate mem for mergeheap

  // *****************************************************************
  // *                                                               *
  // * The main loop.  At the outermost level we are looping over    *
  // * levels of the merge tree.  Typically this will be very small, *
  // * e.g. 1-3.  The final merge pass is handled outside the loop.  *
  // * Future extension may want to do something special in the last *
  // * merge                                                         *
  // *                                                               *
  // *****************************************************************

  if(bProgress){
    //compute merge depth, number of passes over data
    treeHeight=(int)ceil(log((double)nRuns)/log((double)mrgArity));
  }
  
  while (nRuns > mrgArity){
    if(bProgress){
      progCount=0;
      cout << "\rMerge pass " << mrgHeight+1 << " of " << treeHeight 
           << " [  0.00\%]" << flush;
    }
    // We are not yet at the top of the merge tree
    // Write merged runs to temporary output streams
    TP_LOG_DEBUG_ID ("Intermediate merge. level="<<mrgHeight);
    // The number of output runs we form after a mrgArity merge
    nRuns = (nRuns + mrgArity - 1)/mrgArity;

    // Distribute the new nRuns evenly across mrgArity (or fewer)
    // output streams
    minRunsPerStream = nRuns/mrgArity;

    // We may have less mrgArity input runs for the last
    // merged output run.
    arity_t mergeRunsInLastOutputRun=(nXtraRuns>0) ? nXtraRuns : mrgArity;

    // The number of extra runs or the number of streams that
    // get one additional run. This is less than mrgArity and
    // it is OK to downcast to an arity_t.
    nXtraRuns = (arity_t) (nRuns - minRunsPerStream*mrgArity);
    tp_assert(nXtraRuns<mrgArity, "Too many extra runs");

    // How many Streams we will create at the next level
    arity_t nOutputStreams = (minRunsPerStream > 0) ? mrgArity : nXtraRuns;

    arity_t nRunsToMerge = mrgArity; // may change for last output run

    // is current merge output the last run on this merge level?
    bool lastOutputRun = false;

    // open the mrgArity Input streams from which to read runs
    for(ii = 0; ii < mrgArity; ii++){
      // Make the input file name
      make_name(working_disk, suffixName[mrgHeight%2], ii, newName);
      // Dynamically allocate the stream
      // We account for these mmBytesPerStream in phase 2
      // (input stream to read from)
      mergeInputStreams[ii] = new AMI_STREAM<T>(newName);
      mergeInputStreams[ii]->seek(0);
    }

    TPIE_OS_OFFSET check_size=0;
    // For each new output stream, fill with merged runs.
    // strange indexing is that so if there are fewer than mrgArity
    // output streams needed, we use the LAST nOutputStreams. This
    // always keeps the one possible short run in the LAST of the
    // mrgArity output streams.
    TP_LOG_DEBUG_ID ("Writing " << nRuns << " runs to " << nOutputStreams
        << " output files.\nEach output file has at least "
        << minRunsPerStream << " runs.");

    for(ii = mrgArity-nOutputStreams; ii < mrgArity; ii++){
      // Make the output file name
      make_name(working_disk, suffixName[(mrgHeight+1)%2], ii, newName);
      // Dynamically allocate the stream
      // We account for these mmBytesPerStream in phase 2
      // (temp merge output stream)
      curOutputRunStream = new AMI_STREAM<T>(newName);

      // How many runs should this stream get?
      // extra runs go in the LAST nXtraRuns streams so that
      // the one short run is always in the LAST output stream
      runsInStream = minRunsPerStream + ((ii >= mrgArity-nXtraRuns)?1:0);
      TP_LOG_DEBUG_ID ("Writing " << runsInStream << " runs to output "
          << " file " << ii);
      for( jj=0; jj < runsInStream; jj++ ) { // For each run in this stream
          // See if this is the last run.
          if( (ii==mrgArity-1) && (jj==runsInStream-1)) {
            lastOutputRun=true;
            nRunsToMerge=mergeRunsInLastOutputRun;
          }
          // Merge runs to curOutputRunStream
          ae = single_merge(mergeInputStreams+mrgArity-nRunsToMerge,
              nRunsToMerge, curOutputRunStream, nItemsPerRun);
          if (ae != AMI_ERROR_NO_ERROR) {
            TP_LOG_FATAL_ID("AMI_single_merge error"<< ae <<" in deep merge");
            return ae;
          }
      } // For each run in this stream

      // Commit new output stream to disk
      TP_LOG_DEBUG_ID ("Wrote " << runsInStream << " runs and "
          << curOutputRunStream->stream_len() << " items to file " << ii);
      check_size+=curOutputRunStream->stream_len();
      curOutputRunStream->persist(PERSIST_PERSISTENT);
      delete curOutputRunStream;
    } // For each new output stream

    tp_assert(check_size==nInputItems, "item count mismatch in merge");
    // All output streams created/filled.
    // Clean up, go up to next level

    // Delete temp input merge streams
    for(ii = 0; ii < mrgArity; ii++){
      mergeInputStreams[ii]->persist(PERSIST_DELETE);
      delete mergeInputStreams[ii];
    }
    // Update run lengths
    nItemsPerRun=mrgArity*nItemsPerRun; //except for maybe last run
    mrgHeight++; // moving up a level
  } // while (nRuns > mrgArity)

  tp_assert( nRuns > 1, "Not enough runs to merge to final output");
  tp_assert( nRuns <= mrgArity, "Too many runs to merge to final output");

  // We are at the last merge phase, write to specified output stream
  // Open up the nRuns final merge streams to merge
  // These runs are packed in the LAST nRuns elements of the array
  TP_LOG_DEBUG_ID ("Final merge. level="<<mrgHeight);
  TP_LOG_DEBUG_ID ("Merge runs left="<<nRuns);
  for(ii = mrgArity-nRuns; ii < mrgArity; ii++){
    // Make the input file name
   make_name(working_disk, suffixName[mrgHeight%2], ii, newName);
    /* Dynamically allocate the stream
       We account for these mmBytesPerStream in phase 2 
       (input stream to read from)
       Put LAST nRuns files in FIRST nRuns spot here
       either one of mergeInputStreams loading or the call to
       single_merge is a little messy. I put the mess here. (abd)
     */
    TP_LOG_DEBUG_ID ("Putting merge stream "<< ii << " in slot "
        << ii-(mrgArity-nRuns));
    mergeInputStreams[ii-(mrgArity-nRuns)] = new AMI_STREAM<T>(newName);
    mergeInputStreams[ii-(mrgArity-nRuns)]->seek(0);
  }

  if(bProgress){
    progCount=0;
    cout << "\rFinal merge pass (" << mrgHeight+1 << " of " << treeHeight 
      << ") [  0.00\%]" << flush;
  }
  // Merge last remaining runs to the output stream.
  // mergeInputStreams is address( address (the first input stream) )
  ae = single_merge (mergeInputStreams, nRuns, outStream );

  tp_assert(outStream->stream_len() == nInputItems, "item count mismatch");

  if (ae != AMI_ERROR_NO_ERROR) {
    TP_LOG_FATAL_ID ("AMI_ERROR " << ae << " returned by single_merge "
        << "in final merge phase");
    return ae;
  }

  TP_LOG_DEBUG_ID ("merge cleanup");
  if(bProgress){cout << endl;} //print newline
  // We are done, except for cleanup. Is anyone still reading this?
  // Delete temp input merge streams
  for(ii = 0; ii < nRuns; ii++){
    mergeInputStreams[ii]->persist(PERSIST_DELETE);
    delete mergeInputStreams[ii];
  }
  // Delete stream ptr arrays
  delete [] mergeInputStreams;
  // Deallocate the merge heap, free up memory
  MergeHeap.deallocate();
  TP_LOG_DEBUG_ID ("Number of passes incl run formation is " <<
      mrgHeight+2 ); 
  TP_LOG_DEBUG_ID ("AMI_partition_and_merge END");
  return AMI_ERROR_NO_ERROR;
}

template<class T, class I, class M>
AMI_err sort_manager<T,I,M>::single_merge( AMI_STREAM < T > **inStreams,
    arity_t arity, AMI_STREAM < T >*outStream, TPIE_OS_OFFSET cutoff=-1 )
{
  arity_t i;
  AMI_err ami_err;

  TPIE_OS_OFFSET* nread = new TPIE_OS_OFFSET[arity];
  TPIE_OS_OFFSET progStep, progTarget; //for progress bar
  
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
        delete[] nread;
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
  if(bProgress){ 
    progStep=(TPIE_OS_OFFSET)(0.0001*nInputItems);
    progTarget=progCount+progStep;
  }
  while (MergeHeap.sizeofheap() > 0) {

    i = MergeHeap.get_min_run_id ();

    if ((ami_err = outStream->write_item (*in_objects[i]))
        != AMI_ERROR_NO_ERROR) {
      delete[] in_objects;
      delete[] nread;
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
          delete[] nread;
          return ami_err;
        }
      }
    }
    if (ami_err == AMI_ERROR_END_OF_STREAM) {
      MergeHeap.delete_min_and_insert ((T *) NULL);
    } else {
      nread[i]++;
      MergeHeap.delete_min_and_insert (in_objects[i]);
      if(bProgress){
        progCount++;
        if(progCount>progTarget){
          progTarget=progCount+progStep;
          cout << "\b\b\b\b\b\b\b\b\b" <<  "[" << setw(6) 
            << setiosflags(ios::fixed) << setprecision(2) 
            << ((1.*progCount)/nInputItems)*100. << "%]" 
            << flush;
        }
      }
    }
  }//while

  //cleanup
  delete [] in_objects;
  delete [] nread;
  if(bProgress){
    cout << "\b\b\b\b\b\b\b\b\b" <<  "[" << setw(6) << setiosflags(ios::fixed) 
         << setprecision(2) << ((1.*progCount)/nInputItems)*100. << "%]" 
         << flush;
  }
  return AMI_ERROR_NO_ERROR;
}


template<class T, class I, class M>
inline void sort_manager<T,I,M>::make_name(char *prepre, char *pre,
    int id, char *dest)
{
  //This buffer must be long enough to hold the
  //largest possible stream id (in decimal)
  //largest ID is at most mrgArity
  char tmparray[6];

  strcpy (dest, prepre);
  strcat (dest, pre);
  sprintf (tmparray, "%d", id);
  strcat (dest, tmparray);
}

// *******************************************************************
// *                                                                 *
// *           The actual AMI_sort calls                             *
// *                                                                 *
// ******************************************************************* 

// A version of AMI_sort that takes an input stream of elements of type
// T, and an output stream, and and uses the < operator to sort
template<class T>
AMI_err AMI_sort(AMI_STREAM<T> *instream, AMI_STREAM<T> *outstream,
                 bool progress=false)
{
  return sort_manager< T, Internal_Sorter_Op<T>, merge_heap_dh_op<T> >
  (Internal_Sorter_Op<T>(), merge_heap_dh_op<T>() ).sort
  (instream, outstream, progress);
}

// A version of AMI_sort that takes an input stream of elements of
// type T, an output stream, and a user-specified comparison
// object. The comparison object "cmp", of (user-defined) class
// represented by CMPR, must have a member function called "compare"
// which is used for sorting the input stream.

template<class T, class CMPR>
AMI_err AMI_sort(AMI_STREAM<T> *instream, AMI_STREAM<T> *outstream,
    CMPR *cmp, bool progress=false)
{
  return sort_manager<T, Internal_Sorter_Obj<T, CMPR>,
         merge_heap_dh_obj<T,CMPR> >( Internal_Sorter_Obj<T, CMPR>(cmp),
             merge_heap_dh_obj<T,CMPR>(cmp) ).sort
           (instream, outstream, progress);
}

// ********************************************************************
// *                                                                  *
// *  These are the versions that keep a heap of pointers to records  *
// *                                                                  *
// ********************************************************************
// A version of AMI_sort that takes an input stream of elements of type
// T, and an output stream, and and uses the < operator to sort

template<class T>
AMI_err AMI_ptr_sort(AMI_STREAM<T> *instream, AMI_STREAM<T> *outstream,
                 bool progress=false)
{
  return sort_manager< T, Internal_Sorter_Op<T>, merge_heap_pdh_op<T> >
    (Internal_Sorter_Op<T>(), merge_heap_pdh_op<T>()).sort
    (instream, outstream, progress);
}

// A version of AMI_sort that takes an input stream of elements of
// type T, an output stream, and a user-specified comparison
// object. The comparison object "cmp", of (user-defined) class
// represented by CMPR, must have a member function called "compare"
// which is used for sorting the input stream.

template<class T, class CMPR>
AMI_err AMI_ptr_sort(AMI_STREAM<T> *instream, AMI_STREAM<T> *outstream,
    CMPR *cmp, bool progress=false)
{
  return sort_manager<T, Internal_Sorter_Obj<T, CMPR>,
         merge_heap_pdh_obj<T,CMPR> >( Internal_Sorter_Obj<T, CMPR>(cmp),
             merge_heap_pdh_obj<T,CMPR>(cmp) ).sort
           (instream, outstream, progress);
}

// ********************************************************************
// *                                                                  *
// *  This version keeps a heap of keys to records                    *
// *                                                                  *
// ********************************************************************
// A version of AMI_sort that takes an input stream of elements of
// type T, an output stream, a key specification, and a user-specified
// comparison object. 

// The key specification consists of an example key, which is used to
// infer the type of the key field. The comparison object "cmp", of
// (user-defined) class represented by CMPR, must have a member
// function called "compare" which is used for sorting the input
// stream, and a member function called "copy" which is used for
// copying the key of type KEY from a record of type T (the type to be
// sorted).

template<class T, class KEY, class CMPR>
AMI_err  AMI_key_sort(AMI_STREAM<T> *instream, AMI_STREAM<T> *outstream,
    KEY dummykey, CMPR *cmp, bool progress=false)
{
  return sort_manager<T, Internal_Sorter_KObj<T, KEY, CMPR>,
        merge_heap_dh_kobj<T,KEY,CMPR> >( Internal_Sorter_KObj<T,KEY,CMPR>(cmp),
             merge_heap_dh_kobj<T,KEY,CMPR>(cmp) ).sort
          (instream, outstream, progress);
}

// ********************************************************************
// *                                                                  *
// * Duplicates of the above versions that only use 2x space and      *
// * overwrite the original input stream                              * 
// *                                                                  *
// ********************************************************************

// A version of AMI_sort that takes an input stream of elements of type
// T, and and uses the < operator to sort
template<class T>
AMI_err AMI_sort(AMI_STREAM<T> *instream, bool progress=false)
{
  return sort_manager< T, Internal_Sorter_Op<T>, merge_heap_dh_op<T> >
  (Internal_Sorter_Op<T>(), merge_heap_dh_op<T>() ).sort
  (instream, progress);
}

// A version of AMI_sort that takes an input stream of elements of
// type T, and a user-specified comparison
// object. The comparison object "cmp", of (user-defined) class
// represented by CMPR, must have a member function called "compare"
// which is used for sorting the input stream.

template<class T, class CMPR>
AMI_err AMI_sort(AMI_STREAM<T> *instream, CMPR *cmp, bool progress=false)
{
  return sort_manager<T, Internal_Sorter_Obj<T, CMPR>,
         merge_heap_dh_obj<T,CMPR> >( Internal_Sorter_Obj<T, CMPR>(cmp),
             merge_heap_dh_obj<T,CMPR>(cmp) ).sort
           (instream, progress);
}

// ********************************************************************
// *                                                                  *
// *  These are the versions that keep a heap of pointers to records  *
// *                                                                  *
// ********************************************************************
// A version of AMI_sort that takes an input stream of elements of type
// T, and and uses the < operator to sort

template<class T>
AMI_err AMI_ptr_sort(AMI_STREAM<T> *instream, bool progress=false)
{
  return sort_manager< T, Internal_Sorter_Op<T>, merge_heap_pdh_op<T> >
    (Internal_Sorter_Op<T>(), merge_heap_pdh_op<T>()).sort
    (instream, progress);
}

// A version of AMI_sort that takes an input stream of elements of
// type T, and a user-specified comparison
// object. The comparison object "cmp", of (user-defined) class
// represented by CMPR, must have a member function called "compare"
// which is used for sorting the input stream.

template<class T, class CMPR>
AMI_err AMI_ptr_sort(AMI_STREAM<T> *instream, CMPR *cmp, bool progress=false)
{
  return sort_manager<T, Internal_Sorter_Obj<T, CMPR>,
         merge_heap_pdh_obj<T,CMPR> >( Internal_Sorter_Obj<T, CMPR>(cmp),
             merge_heap_pdh_obj<T,CMPR>(cmp) ).sort
           (instream, progress);
}

// ********************************************************************
// *                                                                  *
// *  This version keeps a heap of keys to records                    *
// *                                                                  *
// ********************************************************************
// A version of AMI_sort that takes an input stream of elements of
// type T, a key specification, and a user-specified
// comparison object. 

// The key specification consists of an example key, which is used to
// infer the type of the key field. The comparison object "cmp", of
// (user-defined) class represented by CMPR, must have a member
// function called "compare" which is used for sorting the input
// stream, and a member function called "copy" which is used for
// copying the key of type KEY from a record of type T (the type to be
// sorted).

template<class T, class KEY, class CMPR>
AMI_err  AMI_key_sort(AMI_STREAM<T> *instream, KEY dummykey, CMPR *cmp,
    bool progress=false)
{
  return sort_manager<T, Internal_Sorter_KObj<T, KEY, CMPR>,
        merge_heap_dh_kobj<T,KEY,CMPR> >( Internal_Sorter_KObj<T,KEY,CMPR>(cmp),
             merge_heap_dh_kobj<T,KEY,CMPR>(cmp) ).sort
          (instream, progress);
}

/*
DEPRECATED: comparison function sorting
Earlier TPIE versions allowed a sort that used a C-style
comparison function to sort. However, comparison functions cannot be
inlined, so each comparison requires one function call. Given that the
comparison operator < and comparison object classes can be inlined and
have better performance while providing the exact same functionality,
comparison functions have been removed from TPIE. If you can provide us
with a compelling argument on why they should be in here, we may consider
adding them again, but you must demonstrate that comparision functions
can outperform other methods in at least some cases or give an example
were it is impossible to use a comparison operator or comparison object

Sincerely, 
the management
*/

#endif // _AMI_SORT_SINGLE_DH_H 
