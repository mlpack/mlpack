/*
 *  n_point.h
 *  
 *
 *  Created by William March on 2/24/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef N_POINT_H
#define N_POINT_H

#include "matcher.h"
#include "fastlib/fastlib.h"
#include "n_point_nodes.h"
#include "n_point_impl.h"

const fx_entry_doc n_point_entries[] = {
{"total_runtime", FX_TIMER, FX_CUSTOM, NULL, 
"Total time required to compute the n-point correlation.\n"},
{"n_point_time", FX_TIMER, FX_CUSTOM, NULL, 
"Time required for just the n-point computation (after tree building).\n"},
{"num_tuples", FX_RESULT, FX_INT, NULL,
"The number of matching tuples found.\n"},
{"weighted_num_tuples", FX_RESULT, FX_DOUBLE, NULL,
"The sum of the product of the weights over all matching tuples found.\n"},
{"do_naive", FX_PARAM, FX_BOOL, NULL,
 "If true, the algorithm just runs the base case on the entire data set.\n"},
{"do_hybrid", FX_PARAM, FX_BOOL, NULL,
  "If true, uses hybrid expansion pattern, default: false.\n"},
{"leaf_size", FX_PARAM, FX_INT, NULL,
  "Size of leaves in the kd-tree.  Default: 1.\n"},
{"tree_building", FX_TIMER, FX_CUSTOM, NULL,
  "Time to build the kd-tree.\n"},
{"error", FX_PARAM, FX_DOUBLE, NULL,
  "The relative error allowed between the upper and lower bounds.\n"},
FX_ENTRY_DOC_DONE
};

const fx_submodule_doc n_point_submodules[] = {
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc n_point_doc = {
n_point_entries, n_point_submodules,
"Algorithm module for n-point correlation.\n"
};

class NPointTester;


/**
 *
 */
class NPointAlg {

  friend class NPointTester;
  
private:
  
  

  
  ////////////// variables ////////////

  Matrix data_points_;
  Vector data_weights_;
  
  int num_points_;
  int tuple_size_;
  
  Matcher matcher_;
  
  fx_module* mod_;
  
  bool do_naive_;
  bool do_hybrid_;
  
  int num_tuples_;
  double weighted_num_tuples_;
  
  int num_exclusion_prunes_;
  
  NPointNode* tree_;
  ArrayList<index_t> old_from_new_permutation_;
  int leaf_size_;
  
  int upper_bound_;
  int lower_bound_;
  double error_tolerance_;
  
  ////////////// functions ////////////
  
  
  void NodeIndexTraversal_(NPointNode* tree_, index_t* ind) {
    
    if (tree_->is_leaf()) {
      tree_->stat().set_node_index(*ind);
      //printf("Setting leaf %d\n", *ind);
      *ind = *ind + 1;
    }
    else {
     
      // post order traversal preserves nesting relationship
      NodeIndexTraversal_(tree_->left(), ind);
      NodeIndexTraversal_(tree_->right(), ind);
      
      tree_->stat().set_node_index(*ind);
      //printf("Setting node %d\n", *ind);
      *ind = *ind + 1;
      
    }
    
  } // NodeIndexTraversal_
  
  
  int CountTuples_(ArrayList<NPointNode*>& nodes);
  
  /**
   *
   */
  bool PointsViolateSymmetry_(index_t ind1, index_t ind2);
  
  
  /**
   *
   */
  double HybridHeuristic_(ArrayList<NPointNode*>& nodes);

  
  /**
   *
   */
  int HybridExpansion_();

  
  /**
   *
   */
  int CheckNodeList_(ArrayList<NPointNode*>& nodes);

  
  /**
   *
   */
  int BaseCaseHelper_(ArrayList<ArrayList<index_t> >& point_sets,
                                 ArrayList<bool>& permutation_ok,
                                 ArrayList<index_t>& points_in_tuple,
                                 double* weighed_result, int k);
  
  
  /**
   *
   */
  int BaseCase_(ArrayList<ArrayList<index_t> >& point_sets, 
                   double* weighted_result);
  

  /**
   *
   */
  int DepthFirstRecursion_(ArrayList<NPointNode*>& nodes, 
                           index_t previous_split);

  
  
public:
  
  /**
   *
   */
  void Init(const Matrix& data, const Vector& weights, const Matrix& lower_bds, 
            const Matrix& upper_bds, int n, fx_module* mod) {
    
    mod_ = mod;
    
    data_points_.Copy(data);
    data_weights_.Copy(weights);
    
    num_points_ = data_points_.n_cols();
    
    tuple_size_ = n;
    
    matcher_.Init(lower_bds, upper_bds, tuple_size_);
    
    do_naive_ = fx_param_bool(mod_, "do_naive", false);
    do_hybrid_ = fx_param_bool(mod_, "do_hybrid", false);
    
    if (!do_naive_) {
      
      error_tolerance_ = fx_param_double(mod_, "error", 0.01);
      
      leaf_size_ = fx_param_int(mod_, "leaf_size", 1);
      
      if (leaf_size_ <= 0) {
        FATAL("Leaf size must be strictly positive.\n");
      }
      
      fx_timer_start(mod_, "tree_building");
      
      //printf("node string: %s\n", mod_->key);
      
      tree_ = tree::MakeKdTreeMidpoint<NPointNode>(data_points_, leaf_size_, 
                                                   &old_from_new_permutation_, 
                                                   NULL);
      
      //tree_->Print();
      //printf("======\n\n");
      
      //printf("tree built, leaf_size: %d\n", leaf_size_);
      
      index_t ind = 0;
      NodeIndexTraversal_(tree_, &ind);
      
      fx_timer_stop(mod_, "tree_building");
      
      //printf("timer stopped\n");
      
    } // tree building
    
    upper_bound_ = n_point_impl::NChooseR(num_points_, tuple_size_);
    lower_bound_ = 0;
    
    num_tuples_ = 0;
    weighted_num_tuples_ = 0.0;
    
    num_exclusion_prunes_ = 0;
    
    
    
  } // Init()
  
  /**
   *
   */
  void Compute() {
    
    fx_timer_start(mod_, "n_point_time");
    
    if (do_naive_) {
      
      ArrayList<ArrayList<index_t> > point_sets;
      point_sets.Init(tuple_size_);
      for (index_t i = 0; i < tuple_size_; i++) {
        point_sets[i].Init(num_points_);
        
        for (index_t j = 0; j < num_points_; j++) {
          point_sets[i][j] = j;
        } // for j
        
      } // for i
      
      num_tuples_ = BaseCase_(point_sets, &weighted_num_tuples_);
      
    } // do naive
    else if (!do_hybrid_) {
      
      ArrayList<NPointNode*> node_list;
      node_list.Init(tuple_size_);
      
      for (index_t i = 0; i < tuple_size_; i++) {
        
        node_list[i] = tree_;
        
      } // for i

      num_tuples_ = DepthFirstRecursion_(node_list, -1);
      
    } // do depth-first
    else {
      // do hybrid
      
      printf("Doing hybrid expansion.\n\n");
      
      num_tuples_ = HybridExpansion_();
      
    } // do hybrid
    

    fx_timer_stop(mod_, "n_point_time");

    printf("\n====  Number of tuples: %d ====\n\n", num_tuples_);
    
    fx_result_int(mod_, "num_tuples", num_tuples_);
    fx_result_double(mod_, "weighted_num_tuples", weighted_num_tuples_);
    
  } // Compute()
  
  
}; // NPointAlg


#endif 

