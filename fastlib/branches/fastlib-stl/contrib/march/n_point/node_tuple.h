/*
 *  node_tuple.h
 *  
 *
 *  Created by William March on 2/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef NODE_TUPLE_H
#define NODE_TUPLE_H

#include "fastlib/fastlib.h"

namespace npt {
  
  class NodeTuple { 
    
  public:
    
    typedef BinarySpaceTree<DHrectBound<2>, arma::mat> NptNode;
    
    
  private:
    
    arma::Col<NptNode*> node_list_;
    
    index_t tuple_size_;
    
    // distance ranges in the ordering given by the tuple - unsorted
    std::vector<DRange> ranges_;
    
    std::vector<std::pair<double, index_t> > sorted_upper_;
    std::vector<std::pair<double, index_t> > sorted_lower_;
    
    // map the indices of the nodes to the positions of their distances in the
    // sorted list
    arma::Col<index_t> input_to_upper_;
    arma::Col<index_t> input_to_lower_;
    
    // this is the position of the node we should split next
    index_t ind_to_split_;
    
    bool all_leaves_;
    
    
    //////////////// functions ///////////////////
    
    void SplitHelper_();
    
    
    // given which node we're splitting, which elments of the index lists are 
    // no longer valid
    std::vector<index_t>& FindInvalidIndices_();
    
    // after putting in a new node and getting the invalidated indices from
    // FindInvalidIndices, call this to update the distance lists
    void UpdateIndices_(index_t split_ind, 
                        std::vector<index_t>& invalid_indices);
    
    
    
  public:
    
    // constructor - only use this one to make the original node tuple
    // at the start of the algorithm
    // The copy constructor will be used for the others
    NodeTuple(arma::Col<NptNode*>& list_in) : node_list_(list_in) {
      
      tuple_size_ = list_in.n_rows;
      
      ind_to_split_ = -1;
      int split_size = -1;
      all_leaves = true;
      
      for (index_t i = 0; i < tuple_size_; i++) {
        
        all_leaves_ = all_leaves_ ? node_list_(i)->is_leaf() : false;
        
        if (node_list_(i)->count() > split_size 
            && !(node_list_(i)->is_leaf())) {
          split_size_ = node_list_(i)->count();
          ind_to_split_ = i;
        }
        
        for (index_t j = i+1; j < tuple_size_; j++) {
          
          double max_dist_sq = node_list_(i)->bound().MaxDistanceSq(node_list_(j)->bound());
          double min_dist_sq = node_list_(i)->bound().MinDistanceSq(node_list_(j)->bound());
          
          // TODO: is this still correct?  the api is down. . .
          DRange this_range;
          this_range.Init(min_dist_sq, max_dist_sq);
          
          ranges_.PushBack(this_range);
          
          index_t this_ind = ranges_.size();

          sorted_upper_.PushBack(std::pair<double, index_t> (max_dist_sq, 
                                                             this_ind));
          sorted_lower_.PushBack(std::pair<double, index_t> (min_dist_sq, 
                                                             this_ind));
          
        } // for j
        
      } // for i
      
      // now sort the lists that need it
      std::sort(sorted_upper_.begin(), sorted_upper_.end());
      std::sort(sorted_lower_.begin(), sorted_lower_.end());
      
      input_to_upper_.set_size(sorted_upper_.size());
      input_to_lower_.set_size(sorted_upper_.size());
      
      for (index_t i = 0; i < sorted_upper_.size(); i++) {
        
        input_to_upper_(sorted_upper_[i].second) = i;
        input_to_lower_(sorted_lower_[i].second) = i;
        
      } // loop over pairwise distances
      
    } // constructor (init)
    
    // use this constructor to make children in the recursion
    NodeTuple(NodeTuple& parent, bool is_left) : node_list_(parent.node_list()),
    ranges_(parent.ranges()), sorted_upper_(parent.sorted_upper()), 
    sorted_lower_(parent.sorted_lower())
    {
      
      // assuming that the symmetry has already been checked
      if (is_left) {
        node_list_[index_to_split_] = parent.node_list[index_to_split_]->left();
      }
      else {
        node_list_[index_to_split_] = parent.node_list[index_to_split_]->right();        
      }
      
      ind_to_split_ = parent.ind_to_split();
      
      // now, fix the lists
      // don't forget to make the maps back to the inputs
      input_to_upper_(sorted_upper_.size());
      input_to_lower_(sorted_lower_.size());
      
      // Not sure if this works, if not I should just call these outside
      std::vector<index_t> invalid_inds = FindInvalidIndices_();
      
      UpdateIndices_(ind_to_split, invalid_inds);
      
    } // constructor (children)
    
    NPointNode* node_list(index_t i) {
      return node_list_(i);
    }
    
    arma::Col<NPointNode*>& node_list() const {
      return node_list_;
    }
    
    std::vector<std::pair<double, index_t> >& sorted_upper() const {
      return sorted_upper_;
    }
    std::vector<std::pair<double, index_t> > sorted_lower() const {
      return sorted_lower_;
    }
    
    std::vector<DRange>& ranges() const {
      return ranges_;
    }
    
    double upper_bound(index_t i) {
      return sorted_upper_(i).first;
    }
    
    double lower_bound(index_t i) {
      return sorted_lower_(i).first;
    }
  
    bool all_leaves() {
      return all_leaves_;
    }
    
    index_t ind_to_split() const {
      return ind_to_split_;
    }
    
    //void PerformSplit(NodeTuple* left_ptr, NodeTuple* right_ptr);
    
    bool CheckSymmetry(arma::Col<NptNode*> nodes);
    
    
    
    
  }; // class
  
} // namespace


#endif
