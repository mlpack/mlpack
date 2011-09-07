/*
 *  results_tensor.h
 *  
 *
 *  Created by William March on 6/6/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

// this should only keep the results and tools for updating them


namespace npt {

  class ResultsTensor {
    
  private: 
    
    std::vector<int> results_;
    std::vector<double> weighted_results_;
    
    int tensor_rank_;
    std::vector<int> num_bandwidths_;
    int total_num_bandwidths_;
    std::vector<double> bandwidths_;
    
    
    // just sticking this here for now
    int NChooseR_(int N, int R);
    
    
  public:
    
    ResultsTensor(int tuple_size, const std::vector<int>& num_bands, 
                  const std::vector<double>& min_bands,
                  const std::vector<double>& max_bands) :
                 num_bandwidths_(num_bands)
                 
    {
      
      //tensor_rank_ = (tuple_size * (tuple_size - 1)) / 2;
      tensor_rank_ = num_bandwidths_.size();
      
      total_num_bandwidths_ = 0;
      for (index_t i = 0; i < num_bandwidths_.size(); i++) {
        total_num_bandwidths_ += num_bandwidths_[i];
      }
      
      // todo: double check this
      // this only works if they're all the same 
      int result_size = NChooseR_(total_num_bandwidths_ + tuple_size + 1, 
                                  tuple_size);
      
      results_.resize(result_size);
      weighted_results_.resize(result_size);
      
      for (index_t i = 0; i < tensor_rank_; i++) {
        
        double band_step_i = (max_bands[i] - min_bands[i]) 
                                / (double)num_bandwidths_[i];
        
        
        
      } // fill in individual matchers
      
    } // constructor
    
  }; // class
  
} // namespace