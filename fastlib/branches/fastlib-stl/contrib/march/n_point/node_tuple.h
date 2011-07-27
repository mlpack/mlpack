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
    //arma::Mat<index_t> map_upper_to_sorted_;
    //arma::Mat<index_t> map_lower_to_sorted_;
    
    // sorted upper and lower node distances
    std::vector<double> sorted_upper_;
    std::vector<double> sorted_lower_;
    
    int num_random_;
    
    
    
    // this is the position of the node we should split next
    index_t ind_to_split_;
    
    bool all_leaves_;
    
    
    //////////////// functions ///////////////////
    
    //void SplitHelper_();
    
    
    // given which node we're splitting, which elments of the index lists are 
    // no longer valid
    //void FindInvalidIndices_(std::vector<index_t>& inds);
    
    // after putting in a new node and getting the invalidated indices from
    // FindInvalidIndices, call this to update the distance lists
    //void UpdateIndices_(index_t split_ind, 
    //                    std::vector<index_t>& invalid_indices);
    
    void FillInSortedArrays_();

    void UpdateSplitInd_();

    
    
  public:
    
    // constructor - only use this one to make the original node tuple
    // at the start of the algorithm
    // The copy constructor will be used for the others
    NodeTuple(std::vector<NptNode*>& list_in, int num_random) :
    tuple_size_(list_in.size()),
    sorted_upper_((tuple_size_ * (tuple_size_ - 1)) / 2),
    sorted_lower_((tuple_size_ * (tuple_size_ - 1)) / 2),
    upper_bounds_sq_(tuple_size_, tuple_size_),
    lower_bounds_sq_(tuple_size_, tuple_size_),
    num_random_(num_random)
    {
     
      for (index_t i = 0; i < tuple_size_; i++) {
        node_list_.push_back(list_in[i]);
      } 
      
     UpdateSplitInd_();
     
    
     FillInSortedArrays_();
      
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
    NodeTuple(NodeTuple& parent, bool is_left) : tuple_size_(parent.tuple_size()),
    node_list_(parent.get_node_list()),
    sorted_upper_(parent.sorted_upper().size()), 
    sorted_lower_(parent.sorted_lower().size()),
    upper_bounds_sq_(tuple_size_, tuple_size_),
    lower_bounds_sq_(tuple_size_, tuple_size_),
    num_random_(parent.num_random())
    {
      
      ind_to_split_ = parent.ind_to_split();
      
      // assuming that the symmetry has already been checked
      if (is_left) {
        node_list_[ind_to_split_] = parent.node_list(ind_to_split_)->left();
      }
      else {
        node_list_[ind_to_split_] = parent.node_list(ind_to_split_)->right();        
      }
      
      // Not sure if this works, if not I should just call these outside
      /*
      std::vector<index_t> invalid_inds;
      FindInvalidIndices_(invalid_inds);
      
      UpdateIndices_(ind_to_split_, invalid_inds);
      */
      UpdateSplitInd_();
      FillInSortedArrays_();
      
      
    } // constructor (children)
    
    
    const std::vector<NptNode*>& get_node_list() const {
      return node_list_;
    }
    
    
    const std::vector<double>& sorted_upper() const {
      return sorted_upper_;
    }
    const std::vector<double> sorted_lower() const {
      return sorted_lower_;
    }
    
    /*
    const std::vector<index_t>& input_to_upper() const {
      return input_to_upper_;
    }

    const std::vector<index_t>& input_to_lower() const {
      return input_to_lower_;
    }
     
    
    const std::vector<DRange>& ranges() const {
      return ranges_;
    }
     */
    
    double upper_bound(index_t i) {
      return sorted_upper_[i];
    }
    
    double lower_bound(index_t i) {
      return sorted_lower_[i];
    }
    
    double upper_mat(index_t i, index_t j) {
      return upper_bounds_sq_(i,j);
    }

    double lower_mat(index_t i, index_t j) {
      return lower_bounds_sq_(i,j);
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
    
    index_t tuple_size() const {
      return tuple_size_;
    }
    
    int num_random() const {
      return num_random_;
    }
    
    
    

    bool CheckSymmetry(index_t split_ind, bool is_left);

    
  }; // class
  
} // namespace


#endif
