/*
 *  generic_npt_alg.h
 *  
 *
 *  Created by William March on 8/24/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


/*
 *  This class handles the multi-tree recursion using NodeTuples.  It is 
 *  templated by a matcher class, which has access to the data, checks for 
 *  prunes and stores the results. 
 *
 *  Basically, it does very little while the matcher and NodeTuple do most
 *  of the work.
 */


#ifndef GENERIC_NPT_ALG_H
#define GENERIC_NPT_ALG_H


#include "fastlib/fastlib.h"
#include "node_tuple.h"


namespace npt {
  
  template <class TMatcher>
  class GenericNptAlg {
  
  private:
    
    // Matcher owns the data
    TMatcher& matcher_;
    
    // a list of tree roots
    // NOTE: trees_.size() is the number of distinct sets in this computation
    std::vector<NptNode*> trees_;
    
    // how many times should each tree appear in the tuple?
    std::vector<int> multiplicities_;
    
    int tuple_size_;
    
    int num_prunes_;
    int num_base_cases_;
    
    bool do_naive_;
    
    
    
    bool CanPrune_(NodeTuple& nodes);
    
    void BaseCase_(NodeTuple& nodes);
    
    void DepthFirstRecursion_(NodeTuple& nodes);
    
    
    
  public:
    
    GenericNptAlg(std::vector<NptNode*>& trees_in, 
                  std::vector<int>& multiplicities_in,
                  TMatcher& matcher_in, bool do_naive = false) :
    matcher_(matcher_in), trees_(trees_in), multiplicities_(multiplicities_in)
    {
      
      do_naive_ = do_naive;
      
      //mlpack::IO::Info << "generic alg constructor.\n";
      
      num_prunes_ = 0;
      num_base_cases_ = 0;
      
      tuple_size_ = 0;
      for (unsigned int i = 0; i < multiplicities_.size(); i++) {
        
        tuple_size_ += multiplicities_[i];
        
      }
      
    } // constructor
    
    // Ensures that the matcher now contains the correct results
    void Compute();
    
    void PrintStats() {
     
      mlpack::IO::Info << "num_prunes: " << num_prunes_ << "\n";
      mlpack::IO::Info << "num_base_cases: " << num_base_cases_ << "\n";
      
    }
    
  }; // class
  
  
} // namespace


#include "generic_npt_alg_impl.h"

#endif 

