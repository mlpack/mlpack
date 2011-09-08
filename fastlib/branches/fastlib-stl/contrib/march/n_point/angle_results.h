/*
 *  angle_3pt_alg.h
 *  
 *
 *  Created by William March on 7/27/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ANGLE_RESULTS_H
#define ANGLE_RESULTS_H

#include "angle_matcher.h"
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
 */

namespace npt {

  class AngleResults {
    
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
    
    ///////////////////////////////
    
    
    void AddResult_(int region_id, int num_random, 
                    boost::multi_array<int, 2>& partial_result);
    
    
  public:
    
    AngleResults(int num_regions, 
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
    
    // takes in a (variable-sized) list of regions used in the computation
    // along with the number of randoms involved
    // gets the result out of the matcher, and adds it into results_
    // in the correct place
    // note that region_ids.size() + num_random = tuple_size
    void ProcessResults(std::vector<int> region_ids, int num_random,
                        AngleMatcher& matcher);
    
    void PrintResults();
    
    boost::multi_array<4, int>& results() {
      return results_;
    }
    
        
  }; // class

} //namespace


#endif