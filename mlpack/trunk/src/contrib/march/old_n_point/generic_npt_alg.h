/*
 *  generic_npt_alg.h
 *  
 *
 *  Created by William March on 8/24/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
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
    
    std::vector<NptNode*> trees_;
    std::vector<int> multiplicities_;
    
    int tuple_size_;
    
    int num_prunes_;
    int num_base_cases_;
    
    
    
    bool CanPrune_(NodeTuple& nodes);
    
    void BaseCase_(NodeTuple& nodes);
    
    void DepthFirstRecursion_(NodeTuple& nodes);
    
    
    
  public:
    
    GenericNptAlg(std::vector<NptNode*>& trees_in, 
                  std::vector<int>& multiplicities_in,
                  TMatcher& matcher_in) :
    matcher_(matcher_in), trees_(trees_in)
    {
      
      //mlpack::Log::Info << "generic alg constructor.\n";
      
      num_prunes_ = 0;
      num_base_cases_ = 0;
      
      tuple_size_ = 0;
      for (int i = 0; i < multiplicities_.size(); i++) {
        
        tuple_size_ += multiplicities_[i];
        
      }
      
    } // constructor
    
    // Ensures that the matcher now contains the correct results
    void Compute();
    
    void PrintStats() {
     
      mlpack::Log::Info << "num_prunes: " << num_prunes_ << "\n";
      mlpack::Log::Info << "num_base_cases: " << num_base_cases_ << "\n";
      
    }
    
  }; // class
  
  
} // namespace


#include "generic_npt_alg_impl.h"

#endif 

