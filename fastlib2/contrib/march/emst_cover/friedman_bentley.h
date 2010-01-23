/*
 *  friedman_bentley.h
 *  
 *  Implements the single and multi-fragment kd-tree algorithms from Friedman & 
 *  Bentley's 1977 paper.
 *
 *  Created by William March on 1/4/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FRIEDMAN_BENTLEY_H
#define FRIEDMAN_BENTLEY_H

#include "fastlib/fastlib.h"
#include "mlpack/emst/union_find.h"
#include "emst_cover.h"


const fx_entry_doc fb_entries[] = {
{"MST_computation", FX_TIMER, FX_CUSTOM, NULL, 
"Total time required to compute the MST.\n"},
{"total_length", FX_RESULT, FX_DOUBLE, NULL, 
"The total length of the MST.\n"},
{"number_of_points", FX_RESULT, FX_INT, NULL,
"The number of points in the data set.\n"},
{"dimension", FX_RESULT, FX_INT, NULL,
"The dimensionality of the data.\n"},
{"tree_building", FX_TIMER, FX_CUSTOM, NULL,
"Time taken to construct the kd-tree.\n"},
{"leaf_size", FX_PARAM, FX_INT, NULL,
"Size of leaves in the kd-tree, default 1."},
FX_ENTRY_DOC_DONE
};

const fx_submodule_doc fb_submodules[] = {
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc fb_doc = {
fb_entries, fb_submodules,
"Algorithm module for Bentley-Friedman single-fragment method.\n"
};



class FriedmanBentley {
  
private:
  
  class FBStat {
    
  private:
    
    index_t component_membership_;
    
  public:
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      
      if (count == 1) {
       
        component_membership_ = start;
        
      }
      else {
       
        component_membership_ = -1;
        
      }
      
    } // leaf init
    
    void Init(const Matrix& dataset, index_t start, index_t count,
              const FBStat& left_stat, const FBStat& right_stat) {
      
      Init(dataset, start, count);
      
    } // non leaf Init() 
    
    void set_component_membership(index_t membership) {
      component_membership_ = membership;
    }
    
    index_t component_membership() {
      return component_membership_; 
    }
    
    
    
  }; // tree stat
  
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, FBStat> FBTree;
  
  FBTree* tree_;
  ArrayList<EdgePair> edges_;
  
  fx_module* mod_;
  
  index_t number_of_points_;
  index_t number_of_edges_;
  UnionFind connections_;
  Matrix data_points_;
  
  ArrayList<index_t> old_from_new_permutation_;
  
  double total_dist_;
  
  // is this the right type?
  MinHeap<double, index_t> heap_;
  
  ArrayList<index_t> candidate_neighbors_;
  
  /////////////// functions ///////////////////
  
  void AddEdge_(index_t e1, index_t e2, double distance) {
    
    //EdgePair edge;
    DEBUG_ASSERT_MSG((e1 != e2), 
                     "Indices are equal in single fragment.add_edge(%d, %d, %f)\n", 
                     e1, e2, distance);
    
    DEBUG_ASSERT_MSG((distance >= 0.0), 
                     "Negative distance input in single fragment.add_edge(%d, %d, %f)\n", 
                     e1, e2, distance);
    
    if (e1 < e2) {
      edges_[number_of_edges_].Init(e1, e2, distance);
    }
    else {
      edges_[number_of_edges_].Init(e2, e1, distance);
    }
    
    number_of_edges_++;
    //total_dist_ += distance;
    
  } // AddEdge_

  
  void FindNeighbor_(FBTree* tree, const Vector& point, index_t point_index, 
                     index_t* cand, double* dist);
  
  void UpdateMemberships_(FBTree* tree, index_t point_index);
  
  
  struct SortEdgesHelper_ {
    bool operator() (const EdgePair& pairA, const EdgePair& pairB) {
      return (pairA.distance() > pairB.distance());
    }
  } SortFun;
  
  void SortEdges_() {
    
    std::sort(edges_.begin(), edges_.end(), SortFun);
    
  } // SortEdges_()
  
  
  void EmitResults_(Matrix* results) {
    
    SortEdges_();
    
    DEBUG_ASSERT(number_of_edges_ == number_of_points_ - 1);
    results->Init(3, number_of_edges_);
    
    for (index_t i = 0; i < (number_of_points_ - 1); i++) {
      
      edges_[i].set_lesser_index(old_from_new_permutation_[edges_[i]
                                                           .lesser_index()]);
      
      edges_[i].set_greater_index(old_from_new_permutation_[edges_[i]
                                                            .greater_index()]);
      
      results->set(0, i, edges_[i].lesser_index());
      results->set(1, i, edges_[i].greater_index());
      results->set(2, i, sqrt(edges_[i].distance()));
      total_dist_ += results->get(2, i);
      
    }
    
    fx_result_double(mod_, "total_length", total_dist_);
    
  } // EmitResults_
  
public:
  
  void Init(const Matrix& data, fx_module* mod) {
    
    number_of_edges_ = 0;
    data_points_.Copy(data);
    mod_ = mod;
    
    fx_timer_start(mod_, "tree_building");
    
    index_t leaf_size = fx_param_int(mod_, "leaf_size", 1);
    
    tree_ = tree::MakeKdTreeMidpoint<FBTree>
    (data_points_, leaf_size, &old_from_new_permutation_, NULL);
        
    fx_timer_stop(mod_, "tree_building");
    
    number_of_points_ = data_points_.n_cols();
    edges_.Init(number_of_points_ - 1);
    connections_.Init(number_of_points_);
    
    total_dist_ = 0.0;
    
    candidate_neighbors_.Init(number_of_points_);
    //candidate_neighbors_.SetAll(-1);
    
    heap_.Init();
    
    
  } // Init()
  
  
  void ComputeMST(Matrix* results);
  
  
}; // class





#endif