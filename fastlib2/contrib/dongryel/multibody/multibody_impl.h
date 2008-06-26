#ifndef INSIDE_MULTIBODY_H
#error "This is not a public header file!"
#endif

#ifndef MULTIBODY_IMPL_H
#define MULTIBODY_IMPL_H


template<typename TMultibodyKernel>
int MultitreeMultibody<TMultibodyKernel>::as_indexes_strictly_surround_bs
(Tree *a, Tree *b) {

  return (a->begin() < b->begin() && a->end() >= b->end()) ||
    (a->begin() <= b->begin() && a->end() > b->end());
}

template<typename TMultibodyKernel>
double MultitreeMultibody<TMultibodyKernel>::ttn(int b, 
						 ArrayList<Tree *> &nodes) {
      
  Tree *bkn = nodes[b];
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
      Tree *knj = nodes[j];
      
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
	Tree *knj = nodes[j];
	if(bkn->begin() != knj->begin() ||
	   bkn->end() - 1 != knj->end() - 1) {
	  jdiff = j;
	}
      }
      
      if(jdiff < 0) {
	result = math::BinomialCoefficient(bkn->count(), n - b);
      }
      else {
	Tree *dkn = nodes[jdiff];
	
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

template<typename TMultibodyKernel>
double MultitreeMultibody<TMultibodyKernel>::two_ttn
(int b, ArrayList<Tree *> &nodes, int i) {

  double result = 0.0;
  Tree *kni = nodes[i];
  nodes[i] = kni->left();
  result += ttn(b, nodes);
  nodes[i] = kni->right();
  result += ttn(b, nodes);
  nodes[i] = kni;
  return result;
}

template<typename TMultibodyKernel>
int MultitreeMultibody<TMultibodyKernel>::FindSplitNode
(ArrayList<Tree *> &nodes) {
  
  int global_index = -1;
  int global_min = 0;
  
  for(index_t i = 0; i < non_leaf_indices_.size(); i++) {
    
    /*
      int non_leaf_index = non_leaf_indices_[i];
      double minimum_side_length = MAXDOUBLE;
      
      // find out the minimum side length
      for(index_t j = 0; j < data_.n_rows(); j++) {
      
      DRange range = nodes[non_leaf_index]->bound().get(j);
      double side_length = range.width();
      
      if(side_length < minimum_side_length) {
      minimum_side_length = side_length;
      }
      }
      if(minimum_side_length > global_min) {
      global_min = minimum_side_length;
      global_index = non_leaf_index;
      }
    */
    int non_leaf_index = non_leaf_indices_[i];
    if(nodes[non_leaf_index]->count() > global_min) {
      global_min = nodes[non_leaf_index]->count();
      global_index = non_leaf_index;
    }
  }
  return global_index;
}

template<typename TMultibodyKernel>
bool MultitreeMultibody<TMultibodyKernel>::Prunable
(ArrayList<Tree *> &nodes, double num_tuples, double *allowed_err) {

  return 0;
}

template<typename TMultibodyKernel>
void MultitreeMultibody<TMultibodyKernel>::AddPostponed
(Tree *source_node, Tree *destination_node) {

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

template<typename TMultibodyKernel>
void MultitreeMultibody<TMultibodyKernel>::AddPostponed(Tree *node,
							index_t destination) {
  
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

template<typename TMultibodyKernel>
void MultitreeMultibody<TMultibodyKernel>::MTMultibodyBase
(const ArrayList<Tree *> &nodes, int level) {
  
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
  else {

    // Incorporate postponed force contribution for the given triple
    // of atoms.
    for(index_t i = 0; i < nodes.size(); i++) {
      AddPostponed(nodes[i], exhaustive_indices_[i]);
    }

    // Complete the contribution among three atoms.
    mkernel_.Eval(data_, exhaustive_indices_,
		  negative_force1_e_, negative_force1_u_,
		  positive_force1_l_, positive_force1_e_,
		  negative_force2_e_, negative_force2_u_,
		  positive_force2_l_, positive_force2_e_);
  }

  // Clear all postponed force contribution after incorporating.
  if(level == 0) {
    for(index_t i = 0; i < nodes.size(); i++) {
      nodes[i]->stat().SetZero();
    }
  }
}

template<typename TMultibodyKernel>
void MultitreeMultibody<TMultibodyKernel>::PostProcess(Tree *node) {

  // 
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

template<typename TMultibodyKernel>
void MultitreeMultibody<TMultibodyKernel>::MTMultibody
(ArrayList<Tree *> &nodes, double num_tuples) {
    
  double allowed_err = 0;
  
  /*
    if(Prunable(nodes, num_tuples, &allowed_err)) {
    return;
    }
  */
  
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
    ArrayList<Tree *> new_nodes;
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
  }
}

#endif
