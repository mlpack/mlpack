/*
 *  multi_fragment.h
 *  
 *  Implementation of the NEKMA algorithm of Nevalainen, Ernvall, & Katajainen
 *  (1981).  
 *
 *  Created by William March on 1/19/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef MULTI_FRAGMENT_H
#define MULTI_FRAGMENT_H

#include "fastlib/fastlib.h"
#include "mlpack/emst/union_find.h"
#include "emst_cover.h"

const fx_entry_doc mf_entries[] = {
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

const fx_submodule_doc mf_submodules[] = {
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc mf_doc = {
mf_entries, mf_submodules,
"Algorithm module for NEKMA multi-fragment method.\n"
};


class MultiFragment {
  
private:
  
  class MFStat {
    
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
              const MFStat& left_stat, const MFStat& right_stat) {
      
      Init(dataset, start, count);
      
    } // non leaf Init() 
    
    void set_component_membership(index_t membership) {
      component_membership_ = membership;
    }
    
    index_t component_membership() {
      return component_membership_; 
    }
    
    
  }; // class MFStat

  typedef BinarySpaceTree<DHrectBound<2>, Matrix, MFStat> MFTree;

  class CandidateEdge {
    
  private:
    
    index_t point_in_;
    index_t point_out_;
    //double dist_;
    
  public:

    index_t point_in() {
      return point_in_;
    }
    
    index_t point_out() {
      return point_out_; 
    }
    
    /*
    double dist() {
      return dist_;
    }
     */
    
    bool is_valid(UnionFind& connect) {
      return (connect.Find(point_in_) != connect.Find(point_out_)); 
    }
    
    void Init(index_t pt_in, index_t pt_out) {
      point_in_ = pt_in;
      point_out_ = pt_out;
      //dist_ = distance;
    }
    
  }; // class CandidateEdge
  
  
  /////////// Variables //////////////
  
  MFTree* tree_;
  ArrayList<EdgePair> edges_;
  fx_module* mod_;
  index_t number_of_points_;
  index_t number_of_edges_;
  UnionFind connections_;
  Matrix data_points_;
  
  ArrayList<index_t> old_from_new_permutation_;
  
  double total_dist_;
  
  // keeps the smallest fragment
  MinHeap<index_t, index_t> fragment_size_heap_;
  // fragment_sizes_[i] is the number of points in fragment i
  // is -1 if i is no longer a valid fragment
  ArrayList<index_t> fragment_sizes_;
  
  ArrayList<MinHeap<double, CandidateEdge> > fragment_queues_;
  
  //ArrayList<index_t> candidate_neighbors_;
  
  void AddEdge_(index_t e1, index_t e2, double distance) {
    
    DEBUG_ASSERT(connections_.Find(e1) != connections_.Find(e2));
    
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
  
  
  void FindNeighbor_(MFTree* tree, const Vector& point, index_t point_index, 
                     index_t* cand, double* dist);
  
  void UpdateMemberships_(MFTree* tree, index_t point_index);
  
  
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
  
  void MergeQueues_(index_t comp1, index_t comp2);

  
public:
  
  void Init(const Matrix& data, fx_module* mod) {
    
    number_of_edges_ = 0;
    data_points_.Copy(data);
    mod_ = mod;
    
    fx_timer_start(mod_, "tree_building");
    
    index_t leaf_size = fx_param_int(mod_, "leaf_size", 1);
    
    tree_ = tree::MakeKdTreeMidpoint<MFTree>
    (data_points_, leaf_size, &old_from_new_permutation_, NULL);
    
    fx_timer_stop(mod_, "tree_building");
    
    number_of_points_ = data_points_.n_cols();
    edges_.Init(number_of_points_ - 1);
    connections_.Init(number_of_points_);
    
    total_dist_ = 0.0;
    
    //candidate_neighbors_.Init(number_of_points_);
    //candidate_neighbors_.SetAll(-1);
    
  
    fragment_size_heap_.Init();
    fragment_sizes_.Init(number_of_points_);
    fragment_queues_.Init(number_of_points_);
    
    for (index_t i = 0; i < number_of_points_; i++) {
      
      fragment_size_heap_.Put(1, i);
      fragment_sizes_[i] = 1;
    
      fragment_queues_[i].Init();
      CandidateEdge edge;
      edge.Init(i, i);
      fragment_queues_[i].Put(0.0, edge);
      
      //candidate_neighbors_[i] = i;
      
    } // i
    
    
    
    
  } // Init()
  
  
  void ComputeMST(Matrix* results);
  
  
  
  
}; 




#endif

