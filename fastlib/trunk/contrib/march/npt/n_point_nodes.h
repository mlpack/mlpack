/*
 *  n_point_nodes.h
 *  
 *
 *  Created by William March on 4/28/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef N_POINT_NODES_H
#define N_POINT_NODES_H

#include "fastlib/fastlib.h"  
#include "n_point_impl.h"

class NPointTester;




/**
 *
 */
class NPointStat {
    
private:
  
  index_t node_index_;
  
public:
  
  void Init(const Matrix& dataset, index_t start, index_t count) {
    
    node_index_ = -1;
    
  } // Init() leaves
  
  void Init(const Matrix& dataset, index_t start, index_t count,
            const NPointStat& left_stat, const NPointStat& right_stat) {
    
    node_index_ = -1;
    
  } // Init() non-leaves
  
  index_t node_index() const {
    return node_index_; 
  }
  
  void set_node_index(index_t ind) {
    node_index_ = ind;
  }
  
}; // NPointStat

typedef BinarySpaceTree<DHrectBound<2>, Matrix, NPointStat> NPointNode;


/**
 * 
 */
class NodeTuple {
  
  friend class NPointTester;
  
private:
  
  ArrayList<NPointNode*> node_list_;
  
  int tuple_size_;

  // the distance ranges, in the order specified by the tuple (i.e. unsorted)
  ArrayList<DRange> ranges_;
  
  // sorted distances
  ArrayList<std::pair<double, index_t> > sorted_upper_;
  ArrayList<std::pair<double, index_t> > sorted_lower_;
  
  ArrayList<index_t> input_to_upper_;
  ArrayList<index_t> input_to_lower_;
  
  index_t ind_to_split_;
  bool all_leaves_;
  
  
  ////////////////// functions ///////////////////////
  

  /**
   *
   */
  void FindInvalidIndices_();
  

  
  /**
   *
   */
  void SplitHelper_(ArrayList<NPointNode*>& node_list, 
                    index_t split_ind,
                    ArrayList<DRange>& ranges_in,
                    ArrayList<std::pair<double, index_t> >& upper_in,
                    ArrayList<std::pair<double, index_t> >& lower_in,
                    ArrayList<index_t>& invalid_indices,
                    ArrayList<index_t>& in_to_hi,
                    ArrayList<index_t>& in_to_lo);
  
  

  /**
   *
   */
  void UpdateIndices_(index_t split_ind, ArrayList<index_t>& invalid_indices);

  
  
public:
  
  NPointNode* node_list(index_t i) {
    
    return node_list_[i];
    
  }
  
  double upper_bound(index_t i) {
    return sorted_upper_[i].first;
  }

  double lower_bound(index_t i) {
    return sorted_lower_[i].first;
  }
  

  bool all_leaves() {
    return all_leaves_;
  }
  
  
  void Init(ArrayList<NPointNode*>& list_in) {
    
    tuple_size_ = list_in.size();
    
    node_list_.InitCopy(list_in);
    
    ranges_.Init();
    sorted_upper_.Init();
    sorted_lower_.Init();
    
    ind_to_split_ = -1;
    int split_size = -1;
    
    all_leaves_ = true;
    
    for (index_t i = 0; i < tuple_size_; i++) {
      
      NPointNode* node_i = node_list_[i];
      
      if (!(node_i->is_leaf())) {
        all_leaves_ = false;
      }
      
      if (node_i->count() > split_size && !node_i->is_leaf()) {
        split_size = node_i->count();
        ind_to_split_ = i;
      }
      
      for (index_t j = i+1; j < tuple_size_; j++) {
        
        NPointNode* node_j = node_list_[j];
        
        double min_dist_sq = node_i->bound().MinDistanceSq(node_j->bound());
        double max_dist_sq = node_i->bound().MaxDistanceSq(node_j->bound());
        
        DRange this_range;
        this_range.Init(min_dist_sq, max_dist_sq);
        
        ranges_.PushBackCopy(this_range);
        
        index_t this_ind = sorted_upper_.size();
        sorted_upper_.PushBackCopy(std::pair<double, index_t> (max_dist_sq, 
                                                               this_ind));
        sorted_lower_.PushBackCopy(std::pair<double, index_t>(min_dist_sq, 
                                                              this_ind));
        
      } // for j
      
    } // for i
    
    // sort the indices
    std::sort(sorted_upper_.begin(), sorted_upper_.end());
    std::sort(sorted_lower_.begin(), sorted_lower_.end());
    
    input_to_upper_.Init(sorted_upper_.size());
    input_to_lower_.Init(sorted_lower_.size());
    for (index_t i = 0; i < sorted_upper_.size(); i++) {
      
      input_to_upper_[sorted_upper_[i].second] = i;
      input_to_lower_[sorted_lower_[i].second] = i;
      
    } // fill in input to sorted arrays
  
  } // Init()
  
  void Create();
  
  /**
   * Splits the node in position ind_to_split and updates the lists
   */
  void PerformSplit(NodeTuple*& left_node, NodeTuple*& right_node, 
                    ArrayList<ArrayList<index_t> >& invalid_index_list);
  
  void Print() {
    
    /*
    ArrayList<NPointNode*> node_list_;
    
    int tuple_size_;
    
    // the distance ranges, in the order specified by the tuple (i.e. unsorted)
    ArrayList<DRange> ranges_;
    
    // sorted distances
    ArrayList<std::pair<double, index_t> > sorted_upper_;
    ArrayList<std::pair<double, index_t> > sorted_lower_;
    
    ArrayList<index_t> input_to_upper_;
    ArrayList<index_t> input_to_lower_;
    
    index_t ind_to_split_;
    bool all_leaves_;
*/
    //printf("Node List:\n");
    //ot::Print(node_list_);
    
    printf("Sorted minimum distances:\n");
    for (index_t i = 0; i < sorted_lower_.size(); i++) {
      printf("%g, ", sorted_lower_[i].first);
    }
    printf("\n\n");
    
  } // Print()
  
}; // NodeTuple

#endif




