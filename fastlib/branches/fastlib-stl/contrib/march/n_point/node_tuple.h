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

  typedef BinarySpaceTree<DHrectBound<2>, arma::mat> NptNode;
  
  class NodeTuple { 
    
  public:
    
    
  private:
    
    std::vector<NptNode*> node_list_;
    
    index_t tuple_size_;
    
    // entry i, j is the distance bound for entries i and j in the list above
    arma::mat upper_bounds_sq_;
    arma::mat lower_bounds_sq_;
    
    // IMPORTANT: all the upper and lower bounds start off the same, 
    // so this doesn't matter
    arma::Mat<index_t> map_upper_to_sorted_;
    arma::Mat<index_t> map_lower_to_sorted_;
    
    // sorted upper and lower node distances
    std::vector<double> sorted_upper_;
    std::vector<double> sorted_lower_;
    
    
    
    // this is the position of the node we should split next
    index_t ind_to_split_;
    
    bool all_leaves_;
    
    
    //////////////// functions ///////////////////
    
    void SplitHelper_();
    
    
    // given which node we're splitting, which elments of the index lists are 
    // no longer valid
    void FindInvalidIndices_(std::vector<index_t>& inds);
    
    // after putting in a new node and getting the invalidated indices from
    // FindInvalidIndices, call this to update the distance lists
    void UpdateIndices_(index_t split_ind, 
                        std::vector<index_t>& invalid_indices);
    
    
    
  public:
    
       // constructor - only use this one to make the original node tuple
    // at the start of the algorithm
    // The copy constructor will be used for the others
    NodeTuple(std::vector<NptNode*>& list_in) : 
    upper_bounds_sq_(list_in.size()),
    lower_bounds_sq_(list_in.size(), list_in.size()) 
    {
      
      tuple_size_ = list_in.size();
      
      for (index_t i = 0; i < tuple_size_; i++) {
        node_list_.push_back(list_in[i]);
      } 
      
      ind_to_split_ = -1;
      int split_size = -1;
      all_leaves_ = true;
      
      
      //printf("Checking for leaves\n");
      
      
      for (index_t i = 0; i < tuple_size_; i++) {
        
        if (!(node_list_[i]->is_leaf())) {
          all_leaves_ = false;
          if (node_list_[i]->count() > split_size) {
            split_size = node_list_[i]->count();
            ind_to_split_ = i;
          }
              
        }
        
        
        //printf("Checked for leaves\n");
        
        for (index_t j = i+1; j < tuple_size_; j++) {
          
          double max_dist_sq = node_list_[i]->bound().MaxDistanceSq(node_list_[j]->bound());
          double min_dist_sq = node_list_[i]->bound().MinDistanceSq(node_list_[j]->bound());
          
          upper_bounds_sq_(i, j) = max_dist_sq;
          upper_bounds_sq_(j, i) = max_dist_sq;

          lower_bounds_sq_(i, j) = min_dist_sq;
          lower_bounds_sq_(i, j) = min_dist_sq;
          
          sorted_upper_.push_back(max_dist_sq);
          sorted_lower_.push_back(min_dist_sq);
          
        } // for j
        
      } // for i
      
      //printf("Finished with double loop\n");
      
      // now sort the lists that need it
      std::sort(sorted_upper_.begin(), sorted_upper_.end());
      std::sort(sorted_lower_.begin(), sorted_lower_.end());
      
      /*
      for (index_t i = 0; i < sorted_upper_.size(); i++) {
        std::cout << "sorted_upper[" << i << "]: (";
        std::cout << sorted_upper_[i].first << ", " << sorted_upper_[i].second;
        std::cout << ")\n";
      } 

      std::cout <<"\n";
      
      for (index_t i = 0; i < sorted_lower_.size(); i++) {
        std::cout << "sorted_lower[" << i << "]: (";
        std::cout << sorted_lower_[i].first << ", " << sorted_lower_[i].second;
        std::cout << ")\n";
      } 
       */
      
      //printf("Finished\n");
      
    } // constructor (init)
    
    // use this constructor to make children in the recursion
    NodeTuple(NodeTuple& parent, bool is_left) : node_list_(parent.get_node_list()),
    sorted_upper_(parent.sorted_upper()), 
    sorted_lower_(parent.sorted_lower())
    {
      
      ind_to_split_ = parent.ind_to_split();
      tuple_size_ = node_list_.size();
      
      // assuming that the symmetry has already been checked
      if (is_left) {
        node_list_[ind_to_split_] = parent.node_list(ind_to_split_)->left();
      }
      else {
        node_list_[ind_to_split_] = parent.node_list(ind_to_split_)->right();        
      }
      
      // Not sure if this works, if not I should just call these outside
      std::vector<index_t> invalid_inds;
      FindInvalidIndices_(invalid_inds);
      
      UpdateIndices_(ind_to_split_, invalid_inds);
      
    } // constructor (children)
    
    
    const std::vector<NptNode*>& get_node_list() const {
      return node_list_;
    }
    
    const std::vector<std::pair<double, index_t> >& sorted_upper() const {
      return sorted_upper_;
    }
    const std::vector<std::pair<double, index_t> > sorted_lower() const {
      return sorted_lower_;
    }
    
    const std::vector<index_t>& input_to_upper() const {
      return input_to_upper_;
    }

    const std::vector<index_t>& input_to_lower() const {
      return input_to_lower_;
    }
    
    const std::vector<DRange>& ranges() const {
      return ranges_;
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
    
    index_t ind_to_split() const {
      return ind_to_split_;
    }
    
    NptNode* node_list(index_t i) {
      return node_list_[i];
    }
    
    
    

    bool CheckSymmetry(const std::vector<NptNode*>& nodes, int split_ind, 
                       bool is_left);
        
    
  }; // class
  
} // namespace


#endif
