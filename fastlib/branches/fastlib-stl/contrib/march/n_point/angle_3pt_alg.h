/*
 *  angle_3pt_alg.h
 *  
 *
 *  Created by William March on 7/27/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef ANGLE_3PT_ALG_H
#define ANGLE_3PT_ALG_H

#include "angle_matcher.h"
#include "node_tuple.h"

// Maybe this should take a min, max, step size in theta?

namespace npt {

  class Angle3ptAlg {
    
  private:
    
    int tuple_size_;
    int num_random_;
    
    bool i_is_random_;
    bool j_is_random_;
    bool k_is_random_;
    
    arma::mat data_mat_;
    arma::mat random_mat_;
    
    arma::colvec data_weights_;
    arma::colvec random_weights_;
    
    //arma::Col<index_t> old_from_new_index_data_;
    //arma::Col<index_t> old_from_new_index_random_;
    std::vector<index_t> old_from_new_index_data_;
    std::vector<index_t> old_from_new_index_random_;
    
    // indexed by [num_random][r1][theta]
    std::vector<std::vector<std::vector<int> > > results_;
    std::vector<std::vector<std::vector<double> > > weighted_results_;
    
    std::vector<double> r1_;
    double r2_multiplier_;
    std::vector<double> thetas_;
    
    
    AngleMatcher matcher_;
    
    NptNode* data_tree_;
    NptNode* random_tree_;
    
    int num_prunes_;
    int num_base_cases_;
    
    int leaf_size_;
    
    ///////////////////////////////
    
    
    void BaseCase_(std::vector<NptNode*>& node_list);
    
    bool CanPrune_(std::vector<NptNode*>& node_list);
    
    bool CheckSymmetry_(std::vector<NptNode*>& node_list);
    
    void DepthFirstRecursion_(std::vector<NptNode*>& node_list);
    
    
    
  public:
    
    Angle3ptAlg(arma::mat& data, arma::colvec& weights, 
                arma::mat& rdata, arma::colvec& rweights, int leaf_size,
                std::vector<double>& r1_vec, double r2_mult, 
                std::vector<double>& theta_vec, double bin_fac) :
    matcher_(r1_vec, r2_mult, theta_vec, bin_fac),
    data_mat_(data), random_mat_(rdata), data_weights_(weights), 
    random_weights_(rweights), results_(4), 
    weighted_results_(4), r1_(r1_vec), r2_multiplier_(r2_mult),
    thetas_(theta_vec)
    {
      
      tuple_size_ = 3;
      
      for (int i = 0; i <= tuple_size_; i++) {
        
        results_[i].resize(r1_.size());
        weighted_results_[i].resize(r1_.size());
        
        for (int j = 0; j < r1_.size(); j++) {
          
          results_[i][j].resize(thetas_.size(), 0);
          weighted_results_[i][j].resize(thetas_.size(), 0.0);
          
        }
        
      }
      
      num_prunes_ = 0;
      num_base_cases_ = 0;
      leaf_size_ = leaf_size;
      
      //data_tree_ = mlpack::tree::MakeKdTreeMidpoint<NptNode, double>(data_mat_, leaf_size_,
      //                                                       old_from_new_index_data_);
      
      //random_tree_ = mlpack::tree::MakeKdTreeMidpoint<NptNode, double>(random_mat_, leaf_size_,
      //                                                         old_from_new_index_random_);

      data_tree_ = new NptNode(data_mat_, leaf_size_, old_from_new_index_data_);
      random_tree_ = new NptNode(random_mat_, leaf_size,
                                 old_from_new_index_random_);
      
      // permute weights
      
    } // constructor
    
    
    
    void Compute() {
    
      for (num_random_ = 0; num_random_ <= tuple_size_; num_random_++) {
        
        std::vector<NptNode*> node_list(tuple_size_);
        
        for (index_t i = 0; i < num_random_; i++) {
          
          node_list[i] = random_tree_;
          
        }
        for (index_t i = num_random_; i < tuple_size_; i++) {
          
          node_list[i] = data_tree_;
          
        }
        
        i_is_random_ = (num_random_ > 0);
        j_is_random_ = (num_random_ > 1);
        k_is_random_ = (num_random_ > 2);
        
        DepthFirstRecursion_(node_list);
        
      }
    
    }
    
    
    void OutputResults();
    
    
    
  }; // class

} //namespace


#endif