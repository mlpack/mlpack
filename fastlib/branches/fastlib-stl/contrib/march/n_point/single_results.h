/*
 *  single_results.h
 *  
 *
 *  Created by William March on 9/7/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SINGLE_RESULTS_H
#define SINGLE_RESULTS_H

#include "single_matcher.h"
#include "boost/multi_array.h"

/*
 *  Knows the structure of the results and processes the intermediate 
 *  results from the matcher
 *  This is where a result from n tree nodes gets processed into the 
 *  correct jackknife results
 * 
 *  The generic resampling class handles giving the right stuff to the matcher
 *  and running the algorithm (through the generic algorithm).
 *  This takes the matcher and info on which computation(s) were run and 
 *  puts the results in the right place.
 *
 *  IMPORTANT: for now, only writing this for the angle matchers
 *  In order to do general n, I'll need to figure out how to do runtime
 *  sized multi arrays
 *  
 */


namespace npt {
  
  class SingleResults {

  private:
    
    // indexed by [resampling_region][num_random][r1][theta]
    boost::multi_array<int, 4> results_;
    boost::multi_array<double, 4> weighted_results_;
    
    int num_regions_;
    int num_r1_;
    int num_theta_;
    
    std::vector<double> r1_vec_;
    std::vector<double> theta_vec_;
    
    static int tuple_size_ = 3;
    
    
    
    
  public:
    
    SingleResults(int num_regions, 
                  std::vector<double>& r1_vec,
                  std::vector<double>& theta_vec) :
    num_regions_(num_regions), num_r1_(r1_vec.size()), 
    num_theta_(theta_vec.size()),
    results_(boost::extents[num_regions][tuple_size_ + 1]
             [r1_vec.size()][theta_vec.size()]),
    weighted_results_(boost::extents[num_regions][tuple_size_ + 1]
                      [r1_vec.size()][theta_vec.size()]),
    r1_vec_(r1_vec), theta_vec_(theta_vec)
    {
    
    } // constructor
    
    
    // problem: the matcher doesn't know which r1 and theta it computed
    void ProcessResults(std::vector<int> region_ids, int num_random,
                        SingleMatcher& matcher,
                        int r1_ind, int theta_ind);
    
    
    
  }; // class
  
} // namespace


#endif