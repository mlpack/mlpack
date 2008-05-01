/** @file naive_ortho_range_search.h
 *
 *  This file contains an implementation of a naive algorithm for
 *  orthogonal range search.
 *
 *  @author Dongryeol Lee (dongryel)
 */

#ifndef NAIVE_ORTHO_RANGE_SEARCH_H
#define NAIVE_ORTHO_RANGE_SEARCH_H

#include "fastlib/fastlib.h"


/** @brief Naive orthogonal range search class.
 *
 *  @code
 *    NaiveOrthoRangeSearch search;
 *    search.Init(dataset);
 *    search.Compute(low_coord_limits, high_coord_limits);
 *
 *    ArrayList<bool> naive_search_results;
 *
 *    // Make sure that the vector is uninitialized before passing.
 *    search.get_results(&naive_search_results);
 *  @endcode
 */
template<typename T>
class NaiveOrthoRangeSearch {
  
  // This class object cannot be copied!
  FORBID_ACCIDENTAL_COPIES(NaiveOrthoRangeSearch);

 private:

  /** @brief The dataset. 
   */
  GenMatrix<T> data_;
  
 public:
  
  ////////// Constructor/Destructor //////////

  /** @brief Constructor which does not do anything.
   */
  NaiveOrthoRangeSearch() {}

  /** @brief Destructor which does not do anything.
   */
  ~NaiveOrthoRangeSearch() {}

  ////////// User-level Functions //////////

  /** @brief Initialize the computation object.
   *
   *  @param data The data used for orthogonal range search.
   */
  void Init(const GenMatrix<T> &data) {

    // copy the incoming data
    data_.StaticCopy(data);
  }

  /** @brief The main computation of naive orthogonal range search.
   *
   *  @param low_coord_limits The lower coordinate range of the search window.
   *  @param high_coord_limits The upper coordinate range of the search
   *                           window.
   *  @param search_results Stores the search results for each search window.
   */
  void Compute(const GenMatrix<T> &low_coord_limits, 
	       const GenMatrix<T> &high_coord_limits,
	       ArrayList<ArrayList<bool> > *search_results) {

    // Allocate the space for holding the search results.
    search_results->Init(low_coord_limits.n_cols());

    // Start the search.
    fx_timer_start(NULL, "naive_search");
    for(index_t j = 0; j < low_coord_limits.n_cols(); j++) {
      (*search_results)[j].Init(data_.n_cols());
      for(index_t i = 0; i < data_.n_cols(); i++) {	
	GenVector<T> pt;
	bool flag = true;
	data_.MakeColumnVector(i, &pt);
	
	// Determine which one of the two cases we have: EXCLUDE, SUBSUME
	// first the EXCLUDE case: when dist is above the upper bound distance
	// of this dimension, or dist is below the lower bound distance of
	// this dimension
	for(index_t d = 0; d < data_.n_rows(); d++) {
	  if(pt[d] < low_coord_limits.get(d, j) || 
	     pt[d] > high_coord_limits.get(d, j)) {
	    flag = false;
	    break;
	  }
	}
	(*search_results)[j][i] = flag;
      }
    }
    fx_timer_stop(NULL, "naive_search");
    
    // Search is now finished.
    
  }

};


#endif
