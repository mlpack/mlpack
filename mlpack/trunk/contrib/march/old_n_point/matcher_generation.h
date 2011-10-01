/*
 *  matcher_generation.h
 *  
 *
 *  Created by William March on 6/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MATCHER_GENERATION_H
#define MATCHER_GENERATION_H

#include "fastlib/fastlib.h"


namespace npt {
 
  class MatcherGenerator {
    
  private:
    
    int tuple_size_;
    
    std::vector<double>& min_bands_;
    std::vector<double>& max_bands_;
    std::vector<int>& num_bands_;
    
    std::vector<arma::mat> matchers_;
    
    std::vector<std::vector<double> > matcher_dists_;
    
    void FillInMatchers_(std::vector<size_t>& matcher_ind, int k);
    
    size_t FindWhichMatcher_(size_t i, size_t j);
    
  public: 
    
    
    MatcherGenerator(std::vector<double>& min_bands,
                     std::vector<double>& max_bands,
                     std::vector<int>& num_bands, int tuple_size) :
    min_bands_(min_bands), max_bands_(max_bands), num_bands_(num_bands) 
    {
      
      tuple_size_ = tuple_size;
      
      matcher_dists_.resize(num_bands.size());
      
      for (size_t i = 0; i < num_bands.size(); i++) {
        
        double band_step = (max_bands[i] - min_bands[i]) / ((double)num_bands[i] - 1.0);
        
        matcher_dists_[i].resize(num_bands[i]);
        
        if (num_bands[i] > 1) {
          for (size_t j = 0; j < num_bands[i]; j++) {
            
            matcher_dists_[i][j] = min_bands[i] + (double)j * band_step;
            
          } // for j
        } // if more than one band
        else {
          
          matcher_dists_[i][0] = min_bands[i];
          
        } // only one band
        
      } // for i
      
      std::vector<size_t> matcher_ind(num_bands.size());
      
      FillInMatchers_(matcher_ind, 0);
      
    } // constructor
    
    arma::mat& matcher(size_t i) {
      return matchers_[i];
    }
    
    int num_matchers() {
      return matchers_.size();
    }
    
    void print() {
      
      for (size_t i = 0; i < matchers_.size(); i++) {
        
        matchers_[i].print("Matcher: ");
        std::cout << "\n";
        
        }
    
    }
    
  }; // class
  
    
  
}




#endif

