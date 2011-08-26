/*
 *  generic_npt_alg_impl.h
 *  
 *
 *  Created by William March on 8/25/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


template<class TMatcher>
bool npt::GenericNptAlg<TMatcher>::CanPrune_(NodeTuple& nodes) {
  
  return !(matcher_.TestNodeTuple(nodes));
  
} // CanPrune


template<class TMatcher>
void npt::GenericNptAlg<TMatcher>::BaseCase_(NodeTuple& nodes) {
  
  matcher_.ComputeBaseCase(nodes);
  
} // BaseCase_


template<class TMatcher>
void npt::GenericNptAlg<TMatcher>::DepthFirstRecursion_(NodeTuple& nodes) {
  
  if (nodes.all_leaves()) {
    
    BaseCase_(nodes);
    num_base_cases_++;
    
  } 
  else if (CanPrune_(nodes)) {
    
    num_prunes_++;
    
  }
  else {
    
    // split nodes and call recursion
    
    // TODO: can I infer something about one check from the other?
    
    // left child
    if (nodes.CheckSymmetry(nodes.ind_to_split(), true)) {
      // do left recursion
      
      mlpack::IO::Info << "recursing\n";
      
      NodeTuple left_child(nodes, true);
      DepthFirstRecursion_(left_child);
      
    }
    // TODO: should I count these
    else {
      mlpack::IO::Info << "symmetry prune\n";
    }
    // right child
    if (nodes.CheckSymmetry(nodes.ind_to_split(), false)) {
      
      mlpack::IO::Info << "recursing\n";

      NodeTuple right_child(nodes, false);
      DepthFirstRecursion_(right_child);
      
    }
    
    else {
      mlpack::IO::Info << "symmetry prune\n";
    }
  
  } // recurse 
  
} // DepthFirstRecursion_


template<class TMatcher>
void npt::GenericNptAlg<TMatcher>::Compute() {
  
  for (num_random_ = 0; num_random_ <= tuple_size_; num_random_) {
    
    std::vector<NptNode*> node_list(tuple_size_);
    
    for (int i = 0; i < num_random_; i++) {
      
      node_list[i] = random_tree_root_;
      
    } // for i
    
    for (int i = num_random_; i < tuple_size_; i++) {
      
      node_list[i] = data_tree_root_;
      
    }
    
    // matcher needs to know num_random_ too to store counts correctly
    NodeTuple nodes(node_list, num_random_);
    matcher_.set_num_random(num_random_);
    
    DepthFirstRecursion_(nodes);
    
  } // num_random_
  
  mlpack::IO::Info << "generic num_base_cases: " << num_base_cases_ << "\n";
  mlpack::IO::Info << "generic num_prunes: " << num_prunes_ << "\n";
  
} // Compute

