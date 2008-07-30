#ifndef INSIDE_MULTIBODY_H
#error "This is not a public header file!"
#endif

#ifndef MULTIBODY_IMPL_H
#define MULTIBODY_IMPL_H

template<typename TMultibodyKernel, typename TTree>
int MultitreeMultibody<TMultibodyKernel, TTree>::
as_indexes_strictly_surround_bs(TTree *a, TTree *b) {

  return (a->begin() < b->begin() && a->end() >= b->end()) ||
    (a->begin() <= b->begin() && a->end() > b->end());
}

template<typename TMultibodyKernel, typename TTree>
double MultitreeMultibody<TMultibodyKernel, TTree>::ttn
(int b, ArrayList<TTree *> &nodes) {
      
  TTree *bkn = nodes[b];
  double result;
  int n = nodes.size();
  
  if(b == n - 1) {
    result = (double) bkn->count();
  }
  else {
    int j;
    int conflict = 0;
    int simple_product = 1;
    
    result = (double) bkn->count();
    
    for(j = b + 1 ; j < n && !conflict; j++) {
      TTree *knj = nodes[j];
      
      if (bkn->begin() >= knj->end() - 1) {
	conflict = 1;
      }
      else if(nodes[j - 1]->end() - 1 > knj->begin()) {
	simple_product = 0;
      }
    }
    
    if(conflict) {
      result = 0.0;
    }
    else if(simple_product) {
      for(j = b + 1; j < n; j++) {
	result *= nodes[j]->count();
      }
    }
    else {
      int jdiff = -1; 
      
      // undefined... will eventually point to the
      // lowest j > b such that nodes[j] is different from
      // bkn	
      for(j = b + 1; jdiff < 0 && j < n; j++) {
	TTree *knj = nodes[j];
	if(bkn->begin() != knj->begin() ||
	   bkn->end() - 1 != knj->end() - 1) {
	  jdiff = j;
	}
      }
      
      if(jdiff < 0) {
	result = math::BinomialCoefficient(bkn->count(), n - b);
      }
      else {
	TTree *dkn = nodes[jdiff];
	
	if(dkn->begin() >= bkn->end() - 1) {
	  result = math::BinomialCoefficient(bkn->count(), jdiff - b);
	  if(result > 0.0) {
	    result *= ttn(jdiff, nodes);
	  }
	}
	else if(as_indexes_strictly_surround_bs(bkn, dkn)) {
	  result = two_ttn(b, nodes, b);
	}
	else if(as_indexes_strictly_surround_bs(dkn, bkn)) {
	  result = two_ttn(b, nodes, jdiff);
	}
      }
    }
  }
  return result;
}

template<typename TMultibodyKernel, typename TTree>
double MultitreeMultibody<TMultibodyKernel, TTree>::two_ttn
(int b, ArrayList<TTree *> &nodes, int i) {

  double result = 0.0;
  TTree *kni = nodes[i];
  nodes[i] = kni->left();
  result += ttn(b, nodes);
  nodes[i] = kni->right();
  result += ttn(b, nodes);
  nodes[i] = kni;
  return result;
}

template<typename TMultibodyKernel, typename TTree>
int MultitreeMultibody<TMultibodyKernel, TTree>::FindSplitNode
(ArrayList<TTree *> &nodes) {
  
  int global_index = -1;
  int global_max = 0;
  
  for(index_t i = 0; i < non_leaf_indices_.size(); i++) {
    int non_leaf_index = non_leaf_indices_[i];
    if(nodes[non_leaf_index]->count() > global_max) {
      global_max = nodes[non_leaf_index]->count();
      global_index = non_leaf_index;
    }
  }
  return global_index;
}

template<typename TMultibodyKernel, typename TTree>
void MultitreeMultibody<TMultibodyKernel, TTree>::RefineStatistics_
(int point_index, TTree *destination_node) {
  
  destination_node->stat().negative_gradient1_u =
    std::max(destination_node->stat().negative_gradient1_u,
	     negative_force1_u_[point_index]);
  destination_node->stat().positive_gradient1_l =
    std::min(destination_node->stat().positive_gradient1_l,
	     positive_force1_l_[point_index]);
  
  for(index_t d = 0; d < negative_force2_e_.n_rows(); d++) {
    destination_node->stat().negative_gradient2_u[d] =
      std::max(destination_node->stat().negative_gradient2_u[d],
	       negative_force2_u_.get(d, point_index));
    destination_node->stat().positive_gradient2_l[d] =
      std::max(destination_node->stat().positive_gradient2_l[d],
	       positive_force2_l_.get(d, point_index));
  }
}

template<typename TMultibodyKernel, typename TTree>
void MultitreeMultibody<TMultibodyKernel, TTree>::RefineStatistics_
(TTree *node) {

  TTree *left_node = node->left();
  TTree *right_node = node->right();

  // Take the left and the right node's bound and combine.
  node->stat().negative_gradient1_u = 
    std::max(left_node->stat().negative_gradient1_u +
	     left_node->stat().postponed_negative_gradient1_u,
	     right_node->stat().negative_gradient1_u +
	     right_node->stat().postponed_negative_gradient1_u);
  node->stat().positive_gradient1_l =
    std::min(left_node->stat().positive_gradient1_l +
	     left_node->stat().postponed_positive_gradient1_l,
	     right_node->stat().positive_gradient1_l +
	     right_node->stat().postponed_positive_gradient1_l);
  
  for(index_t d = 0; d < negative_force2_e_.n_rows(); d++) {
    node->stat().negative_gradient2_u[d] =
      std::max(left_node->stat().negative_gradient2_u[d] +
	       left_node->stat().postponed_negative_gradient2_u[d],
	       right_node->stat().negative_gradient2_u[d] +
	       right_node->stat().postponed_negative_gradient2_u[d]);
    node->stat().positive_gradient2_l[d] =
      std::min(left_node->stat().positive_gradient2_l[d] +
	       left_node->stat().postponed_positive_gradient2_l[d],
	       right_node->stat().positive_gradient2_l[d] +
	       right_node->stat().postponed_positive_gradient2_l[d]);
  }
}

template<typename TMultibodyKernel, typename TTree>
void MultitreeMultibody<TMultibodyKernel, TTree>::AddPostponed
(TTree *source_node, TTree *destination_node) {

  destination_node->stat().postponed_negative_gradient1_e +=
    source_node->stat().postponed_negative_gradient1_e;
  destination_node->stat().postponed_negative_gradient1_u +=
    source_node->stat().postponed_negative_gradient1_u;
  destination_node->stat().postponed_positive_gradient1_l +=
    source_node->stat().postponed_positive_gradient1_l;
  destination_node->stat().postponed_positive_gradient1_e +=
    source_node->stat().postponed_positive_gradient1_e;

  la::AddTo(data_.n_rows(), 
	    source_node->stat().postponed_negative_gradient2_e.ptr(),
	    destination_node->stat().postponed_negative_gradient2_e.ptr());
  la::AddTo(data_.n_rows(), 
	    source_node->stat().postponed_negative_gradient2_u.ptr(),
	    destination_node->stat().postponed_negative_gradient2_u.ptr());
  la::AddTo(data_.n_rows(),
	    source_node->stat().postponed_positive_gradient2_l.ptr(),
	    destination_node->stat().postponed_positive_gradient2_l.ptr());
  la::AddTo(data_.n_rows(),
	    source_node->stat().postponed_positive_gradient2_e.ptr(),
	    destination_node->stat().postponed_positive_gradient2_e.ptr());
}

template<typename TMultibodyKernel, typename TTree>
void MultitreeMultibody<TMultibodyKernel, TTree>::AddPostponed
(TTree *node, index_t destination) {
  
  negative_force1_e_[destination] += 
    node->stat().postponed_negative_gradient1_e;
  negative_force1_u_[destination] +=
    node->stat().postponed_negative_gradient1_u;
  positive_force1_l_[destination] +=
    node->stat().postponed_positive_gradient1_l;
  positive_force1_e_[destination] +=
    node->stat().postponed_positive_gradient1_e;
  la::AddTo(data_.n_rows(), node->stat().postponed_negative_gradient2_e.ptr(),
	    negative_force2_e_.GetColumnPtr(destination));
  la::AddTo(data_.n_rows(), node->stat().postponed_negative_gradient2_u.ptr(),
	    negative_force2_u_.GetColumnPtr(destination));
  la::AddTo(data_.n_rows(), node->stat().postponed_positive_gradient2_l.ptr(),
	    positive_force2_l_.GetColumnPtr(destination));
  la::AddTo(data_.n_rows(), node->stat().postponed_positive_gradient2_e.ptr(),
	    positive_force2_e_.GetColumnPtr(destination));
}

template<typename TMultibodyKernel, typename TTree>
void MultitreeMultibody<TMultibodyKernel, TTree>::MTMultibodyBase
(const ArrayList<TTree *> &nodes, int level) {
  
  int start_index;
  int num_nodes = nodes.size();

  // Recurse to get a $n$ tuple.
  if(level < num_nodes) {
    
    // Run over each point in this node.
    if(level > 0) {
      if(nodes[level - 1] == nodes[level]) {
	start_index = exhaustive_indices_[level - 1] + 1;
      }
      else {
	start_index = nodes[level]->begin();
      }
    }
    else {
      start_index = nodes[level]->begin();
    }
    
    for(index_t i = start_index; i < (nodes[level])->end(); i++) {	
      exhaustive_indices_[level] = i;
      MTMultibodyBase(nodes, level + 1);
    }
  }

  // $n$-tuple is chosen.
  else {

    // Complete the contribution among three atoms.
    mkernel_.Eval(data_, exhaustive_indices_,
		  negative_force1_e_, negative_force1_u_,
		  positive_force1_l_, positive_force1_e_,
		  negative_force2_e_, negative_force2_u_,
		  positive_force2_l_, positive_force2_e_);
  }

  // Add and clear all postponed force contribution after
  // incorporating and refine node statistics.
  if(level == 0) {
    for(index_t i = 0; i < nodes.size(); i++) {

      if(i != 0 && nodes[i] == nodes[i - 1]) {
	continue;
      }

      nodes[i]->stat().negative_gradient1_u = -DBL_MAX;
      nodes[i]->stat().positive_gradient1_l = DBL_MAX;
      nodes[i]->stat().negative_gradient2_u.SetAll(-DBL_MAX);
      nodes[i]->stat().positive_gradient2_l.SetAll(DBL_MAX);

      for(index_t r = nodes[i]->begin(); r < nodes[i]->end(); r++) {
	AddPostponed(nodes[i], r);
	RefineStatistics_(r, nodes[i]);
      }
      nodes[i]->stat().SetZero();
    }
  }
}

template<typename TMultibodyKernel, typename TTree>
void MultitreeMultibody<TMultibodyKernel, TTree>::PostProcess(TTree *node) {

  // For a leaf node,
  if(node->is_leaf()) {
    for(index_t q = node->begin(); q < node->end(); q++) {

      // Add postponed contribution to each point's force vector.
      AddPostponed(node, q);

      // Now, reconstruct the force vector from the complete
      // approximations.
      double *query_total_force_e = total_force_e_.GetColumnPtr(q);
      for(index_t d = 0; d < data_.n_rows(); d++) {

	// First, add in the negative contributions then the positive
	// contributions.
	if(data_.get(d, q) < 0) {
	  query_total_force_e[d] += (-data_.get(d, q) * negative_force1_e_[q] +
				     negative_force2_e_.get(d, q));
	  query_total_force_e[d] += (-data_.get(d, q) * positive_force1_e_[q] +
				     positive_force2_e_.get(d, q));
	}
	else {
	  query_total_force_e[d] += (-data_.get(d, q) * positive_force1_e_[q] +
				     negative_force2_e_.get(d, q));
	  query_total_force_e[d] += (-data_.get(d, q) * negative_force1_e_[q] +
				     positive_force2_e_.get(d, q));
	}
      } // end of iterating over each dimension...
    } // end of iterating over each query point...

    // Clear postponed information.
    node->stat().SetZero();
  }
  else {

    // Push down postponed contributions to the nodes below and clear
    // them.
    AddPostponed(node, node->left());
    AddPostponed(node, node->right());
    node->stat().SetZero();
    
    // Recurse.
    PostProcess(node->left());
    PostProcess(node->right());
  }
}

template<typename TMultibodyKernel, typename TTree>
void MultitreeMultibody<TMultibodyKernel, TTree>::PostProcessNaive_
(TTree *node) {

  for(index_t q = node->begin(); q < node->end(); q++) {
    
    // Add postponed contribution to each point's force vector.
    AddPostponed(node, q);
    
    // Now, reconstruct the force vector from the complete
    // approximations.
    double *query_total_force_e = total_force_e_.GetColumnPtr(q);
    for(index_t d = 0; d < data_.n_rows(); d++) {
      
      // First, add in the negative contributions then the positive
      // contributions.
      if(data_.get(d, q) < 0) {
	query_total_force_e[d] += (-data_.get(d, q) * negative_force1_e_[q] +
				   negative_force2_e_.get(d, q));
	query_total_force_e[d] += (-data_.get(d, q) * positive_force1_e_[q] +
				   positive_force2_e_.get(d, q));
      }
      else {
	query_total_force_e[d] += (-data_.get(d, q) * positive_force1_e_[q] +
				   negative_force2_e_.get(d, q));
	query_total_force_e[d] += (-data_.get(d, q) * negative_force1_e_[q] +
				   positive_force2_e_.get(d, q));
      }
    } // end of iterating over each dimension...
  } // end of iterating over each query point...
  
  // Clear postponed information.
  node->stat().SetZero();
}

template<typename TMultibodyKernel, typename TTree>
bool MultitreeMultibody<TMultibodyKernel, TTree>::Prunable
(ArrayList<TTree *> &nodes, double num_tuples, double *allowed_err) {

  return mkernel_.Eval(nodes, relative_error_, total_n_minus_one_num_tuples_);
}

template<typename TMultibodyKernel, typename TTree>
void MultitreeMultibody<TMultibodyKernel, TTree>::MTMultibody
(ArrayList<TTree *> &nodes, double num_tuples) {
    
  double allowed_err = 0;
  
  if(Prunable(nodes, num_tuples, &allowed_err)) {
    num_prunes_++;
    return;
  }
  
  // Figure out which ones are non-leaves.
  non_leaf_indices_.Resize(0);
  for(index_t i = 0; i < 3; i++) {
    if(!(nodes[i]->is_leaf())) {
      non_leaf_indices_.PushBackCopy(i);
    }
  }
  
  // All leaves, then base case.
  if(non_leaf_indices_.size() == 0) {
    MTMultibodyBase(nodes, 0);
    return;
  }
  
  // Else, split an internal node and recurse.
  else {
    int split_index;
    double new_num_tuples;
    
    // Copy to new nodes list before recursing.
    ArrayList<TTree *> new_nodes;
    new_nodes.Init(mkernel_.order());
    for(index_t i = 0; i < mkernel_.order(); i++) {
      new_nodes[i] = nodes[i];
    }
    
    // Apply splitting heuristic.
    split_index = FindSplitNode(nodes);
    
    // Push down approximations downward for the node that is to be
    // expanded.
    AddPostponed(nodes[split_index], nodes[split_index]->left());
    AddPostponed(nodes[split_index], nodes[split_index]->right());
    nodes[split_index]->stat().SetZero();

    // Recurse to the left.
    new_nodes[split_index] = nodes[split_index]->left();
    new_num_tuples = ttn(0, new_nodes);
    
    // If the current node combination is valid, then recurse.
    if(new_num_tuples > 0) {
      MTMultibody(new_nodes, new_num_tuples);
    }
    
    // Recurse to the right.
    new_nodes[split_index] = nodes[split_index]->right();
    new_num_tuples = ttn(0, new_nodes);
    
    // If the current node combination is valid, then recurse.
    if(new_num_tuples > 0) {
      MTMultibody(new_nodes, new_num_tuples);
    }

    // Refine statistics after recursing.
    RefineStatistics_(nodes[split_index]);
  }
}

#endif
