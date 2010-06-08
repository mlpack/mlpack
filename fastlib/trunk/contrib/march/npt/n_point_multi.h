/*
 *  n_point_multi.h
 *  
 *
 *  Created by William March on 4/14/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef N_POINT_MULTI_H
#define N_POINT_MULTI_H

#include "multi_matcher.h"
#include "fastlib/fastlib.h"
#include "n_point_impl.h"
#include "results_tensor.h"
#include "n_point_nodes.h"

class NPointMulti { 
  
private:
  
    
  int num_bandwidths_;
  
  MultiMatcher matcher_;
  
  ResultsTensor results_;
  
  fx_module* mod_;
  
  Matrix data_points_;
  int leaf_size_;
  
  NPointNode* tree_;
  
  int tuple_size_; 
  int n_choose_2_;
  
  int num_total_prunes_;
  
  ArrayList<ArrayList<index_t> > invalid_indices_;
  
  
  ///////////////// functions ////////////////////
  
  
  bool SymmetryCorrect_(ArrayList<NPointNode*>& nodes);

  index_t CheckBaseCase_(ArrayList<NPointNode*>& nodes);
  
  index_t FindInd_(index_t i, index_t j);

  void BaseCaseHelper_(ArrayList<ArrayList<index_t> >& point_sets,
                       ArrayList<bool>& permutations_ok,
                       ArrayList<index_t>& points_in_tuple, 
                       int k, 
                       ArrayList<GenMatrix<index_t> >& permutation_ranges);
    
  
  void BaseCase_(NodeTuple& nodes, 
                 ArrayList<std::pair<double, double> >& valid_ranges);
  
  void DepthFirstRecursion_(NodeTuple& nodes, 
                            ArrayList<std::pair<double, double> >& valid_ranges);
  
  
  bool PointsViolateSymmetry_(index_t ind1, index_t ind2);
  
  void FindInvalidIndices_();


public:
  
  
  void Init(const Matrix& data, double band_min, double band_max, 
            int num_bands, int n, fx_module* mod) {
    
    mod_ = mod;
    data_points_.Copy(data);
    
    tuple_size_ = n;
    n_choose_2_ = n_point_impl::NChooseR(tuple_size_, 2);
    
    leaf_size_ = fx_param_int(mod_, "leaf_size", 1);
    
    
    // initialize results tensor
    results_.Init(tuple_size_, num_bands);
    
    
    // initialize matcher
    ArrayList<double> dists_sq;
    dists_sq.Init(num_bands);
    
    
    double this_dist = band_min;
    double dist_step = (band_max - band_min) / (double)(num_bands-1);
    
    //printf("dist_step = %g\n", dist_step);
    
    // TODO: double check this
    for (index_t i = 0; i < num_bands; i++) {
      
      dists_sq[i] = this_dist * this_dist;
      this_dist += dist_step;
      
    }
    matcher_.Init(dists_sq, tuple_size_);
    
    ArrayList<index_t> old_from_new;
    tree_ = tree::MakeKdTreeMidpoint<NPointNode> (data_points_, leaf_size_, 
                                                  &old_from_new, NULL);
    
    num_total_prunes_ = 0;
    
    FindInvalidIndices_();
    
  } // Init() 
  
  void Compute() {
    
    fx_timer_start(mod_, "n_point_time");

    NodeTuple nodes;
    ArrayList<NPointNode*> node_list;
    node_list.Init(tuple_size_);
    
    
    for (index_t i = 0; i < tuple_size_; i++) {
      
      node_list[i] = tree_;
      
    } // for i
    
    nodes.Init(node_list);
    
    ArrayList<std::pair<double, double> > valid_ranges;
    valid_ranges.Init(n_choose_2_);
    
    for (index_t i = 0; i < valid_ranges.size(); i++) {
      valid_ranges[i].first = 0.0;
      valid_ranges[i].second = DBL_MAX;
    }
    
    DepthFirstRecursion_(nodes, valid_ranges);
    
    
    fx_timer_stop(mod_, "n_point_time");
    
    fx_result_int(mod_, "num_total_prunes", num_total_prunes_);

    const char* filename = fx_param_str(mod_, "output_file", "output.txt");
    
    FILE* fp;
    fp = fopen(filename, "w");
    
    results_.Output(matcher_.distances(), fp);
    
    fclose(fp);
    
  } // Compute
  
  
}; // NPointMulti


#endif
