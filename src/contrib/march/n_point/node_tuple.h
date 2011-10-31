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

#include <mlpack/core.h>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

/*
 *  This class holds an ordered tuple of n tree nodes (for an n-point 
 *  calculation).  It can be viewed as a node in the recursion tree during 
 *  a computation.  
 *
 *  This class is responsible for the symmetry in the computation.
 *
 *  TODO: think about how to make this more efficient by incorporating threading
 *  Also, look out for overhead from the construction.
 *
 */

using namespace mlpack;

namespace npt {

  typedef tree::BinarySpaceTree<bound::HRectBound<2> > NptNode;
  
  // this is now just responsible for checking symmetry
  class NodeTuple { 
    
  private:
    
    std::vector<NptNode*> node_list_;
    
    std::vector<int> same_nodes_;
    
    int tuple_size_;
    
    
    
    // this is the position of the node we should split next
    size_t ind_to_split_;
    
    bool all_leaves_;
    
    
    //////////////// functions ///////////////////
    
    void UpdateSplitInd_();

    
  public:
    
    // constructor - only use this one to make the original node tuple
    // at the start of the algorithm
    // The copy constructor will be used for the others
    NodeTuple(std::vector<NptNode*>& list_in) :
    node_list_(list_in),
    same_nodes_(list_in.size()), tuple_size_(list_in.size())
    {
     
      same_nodes_[0] = 0;
      for (unsigned int i = 1; i < node_list_.size(); i++) {
        
        if (node_list_[i] == node_list_[i-1]) {
          
          same_nodes_[i] = same_nodes_[i-1];
          
        }
        else {
          
          same_nodes_[i] = 1 + same_nodes_[i-1];
          
        }
        
      }
      
     /*
      for (size_t i = 0; i < tuple_size_; i++) {
        node_list_.push_back(list_in[i]);
      }
      */
      
     UpdateSplitInd_();
      
    } // constructor (init)
    
    // use this constructor to make children in the recursion
    NodeTuple(NodeTuple& parent, bool is_left) : 
    node_list_(parent.get_node_list()),
    same_nodes_(parent.get_same_nodes()),
    tuple_size_(parent.tuple_size())
    {
      
      /*
      mlpack::Log::Info << "constructing child.\n";
      mlpack::Log::Info << "parent: " << parent.node_list(0);
      mlpack::Log::Info << ", " << parent.node_list(1) << ", ";
      mlpack::Log::Info << parent.node_list(2) << "\n";
      
      mlpack::Log::Info << "child (before update): " << node_list_[0];
      mlpack::Log::Info << ", " << parent.node_list_[1] << ", ";
      mlpack::Log::Info << node_list_[2] << "\n\n";
      */
      
      /*
      if (parent.node_list(0)->begin() == 0 &&
          parent.node_list(0)->count() == 48 &&
          parent.node_list(1)->begin() == 0 &&
          parent.node_list(1)->count() == 93 &&
          parent.node_list(2)->begin() == 0 &&
          parent.node_list(2)->count() == 93) {
        
        mlpack::Log::Info << "found it.\n";
        
      }
       */
      
      // TODO: make this more efficient
      // why wasnt the initialization list working?
      /*
      for (int i = 0; i < tuple_size_; i++) {
        
        node_list_[i] = parent.node_list(i);
        
      }
       */
      
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
      std::vector<size_t> invalid_inds;
      FindInvalidIndices_(invalid_inds);
      
      UpdateIndices_(ind_to_split_, invalid_inds);
      */
      
      UpdateSplitInd_();
      
      //FillInSortedArrays_();
      
      
    } // constructor (children)
    
    
    const std::vector<NptNode*>& get_node_list() const {
      return node_list_;
    }
    
    const std::vector<int>& get_same_nodes() const {
      return same_nodes_;
    }
    
    bool all_leaves() {
      return all_leaves_;
    }
    
    size_t ind_to_split() const {
      return ind_to_split_;
    }
    
    NptNode*& node_list(size_t i) {
      return node_list_[i];
    }
    
    size_t tuple_size() const {
      return tuple_size_;
    }
    
    
    
    

    bool CheckSymmetry(size_t split_ind, bool is_left);

    
  }; // class
  
} // namespace


#endif
