#ifndef INSIDE_NWRCDE_H
#error "This is not a public header file!"
#endif

template<typename TKernel>
void NWRCde<TKernel>::NWRCdeBase_(const Matrix &qset, Tree *qnode, Tree *rnode,
				  double probability, 
				  NWRCdeResults &query_results) {

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().ResetBoundStatistics();

  // Compute unnormalized sum for each query point.
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {

    // Incorporate the postponed information.
    NWRCdeCommon::AddPostponed(qnode, q, this);

    // Get the query point.
    const double *q_col = qset_.GetColumnPtr(q);
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // pairwise distance and kernel value
      double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
      double kernel_value = ka_.kernel_.EvalUnnormOnSq(dsqd);
      double weighted_kernel_value = rset_weights_[r] * kernel_value;
      
      query_results.numerator_sum_l[q] += weighted_kernel_value;
      query_results.numerator_sum_e[q] += weighted_kernel_value;
      query_results.denominator_sum_l[q] += kernel_value;
      query_results.denominator_sum_e[q] += kernel_value;

    } // end of iterating over each reference point.
    
    // Each query point has taken care of all reference points.
    query_results.n_pruned_[q] += 
      rnode->stat().farfield_expansion_.get_weight_sum();
    
    // Refine min and max summary statistics.
    NWRCdeCommon::RefineBoundStatistics(q, qnode, this);

  } // end of looping over each query point.

  // Clear postponed information.
  qnode->stat().ClearPostponed();
}

template<typename TKernel>
void NWRCde<TKernel>::NWRCdeCanonical_(const Matrix &qset, Tree *qnode,
				       Tree *rnode, double probability,
				       NWRCdeResults &query_results) {

  // Try finite difference pruning first.
  if(DualtreeKdeCommon::Prunable(qnode, rnode, probability, dsqd_range, 
				 kernel_value_range, dl, de, du, used_error, 
				 n_pruned, this)) {
    num_finite_difference_prunes_++;
    return true;
  }

  // Then Monte Carlo-based pruning.
  else if(probability < 1 &&
	  DualtreeKdeCommon::MonteCarloPrunable_
	  (qnode, rnode, probability, dsqd_range, 
	   kernel_value_range, dl, de, du, used_error, n_pruned, this)) {
    num_monte_carlo_prunes_++;
    return false;
  }

  else if(qset_.n_rows() <= 5 &&
	  PrunableEnhanced_(qnode, rnode, probability, dsqd_range, 
			    kernel_value_range, dl, du, used_error, n_pruned, 
			    order_farfield_to_local, order_farfield, 
			    order_local)) {
    
    // far field to local translation
    if(order_farfield_to_local >= 0) {
      rnode->stat().farfield_expansion_.TranslateToLocal
	(qnode->stat().local_expansion_, order_farfield_to_local);
    }
    // far field pruning
    else if(order_farfield >= 0) {
      for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	densities_e_[q] += 
	  rnode->stat().farfield_expansion_.EvaluateField(qset_, q, 
							  order_farfield);
      }
    }
    // local accumulation pruning
    else if(order_local >= 0) {
      qnode->stat().local_expansion_.AccumulateCoeffs(rset_, rset_weights_,
						      rnode->begin(), 
						      rnode->end(),
						      order_local);
    }
    qnode->stat().postponed_l_ += dl;
    qnode->stat().postponed_u_ += du;
    qnode->stat().postponed_used_error_ += used_error;
    qnode->stat().postponed_n_pruned_ += n_pruned;
    return true;
  }
  
  // For a leaf query node,
  if(qnode->is_leaf()) {
    
    // For leaf pairs, do exhaustive computations.
    if(rnode->is_leaf()) {
      NWRCdeBase_(qset, qnode, rnode, probability, query_results);
      return true;
    }
    
    // For a non-leaf reference, expand reference node,
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
      NWRCdeCommon::BestNodePartners
	(qnode, rnode->left(), rnode->right(), probability,
	 &rnode_first, &probability_first, &rnode_second, &probability_second);
      
      bool first_result = 
	NWRCdeCanonical_(qset, qnode, rnode_first, probability_first,
			 query_results);

      // If the first recursion is computed exactly, then increment
      // the probability tolerance for the second recursion.
      if(first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool second_result =
	NWRCdeCanonical_(qset, qnode, rnode_second, probability_second,
			 query_results);
      return first_result && second_result;
    }
  }
  
  // For a non-leaf query node,
  else {

    // The boolean flag that states that the contribution of the
    // current reference node is computed exactly for the current
    // query node.
    bool result = true;

    // Push down postponed bound changes owned by the current query
    // node to the children of the query node and clear them.
    qnode->left()->stat().AddPostponed(qnode->stat());
    qnode->right()->stat().AddPostponed(qnode->stat());
    
    // Clear out the postponed info after being passed down.
    qnode->stat().ClearPostponed();
    
    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      Tree *qnode_first = NULL, *qnode_second = NULL;
      double probability_first = 0, probability_second = 0;

      NWRCdeCommon::BestNodePartners
	(rnode, qnode->left(), qnode->right(), probability,
	 &qnode_first, &probability_first, &qnode_second, &probability_second);
      bool first_result =
	NWRCdeCanonical_(qset, qnode_first, rnode, probability, query_results);
      bool second_result =
	NWRCdeCanonical_(qset, qnode_second, rnode, probability,
			 query_results);
      result = first_result && second_result;
    }
    
    // For a non-leaf reference node, expand both query and reference
    // nodes.
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
  
      // Fix the query node to be the left child, and recurse.
      NWRCdeCommon::BestNodePartners
	(qnode->left(), rnode->left(), rnode->right(), probability, 
	 &rnode_first, &probability_first, &rnode_second, &probability_second);
      bool left_first_result =
	NWRCdeCanonical_(qset, qnode->left(), rnode_first, probability_first,
			 query_results);

      // If the first recursion is carried out exactly, then increment
      // the probability tolerance for the second recursion.
      if(left_first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool left_second_result = 
	NWRCdeCanonical_(qset, qnode->left(), rnode_second, probability_second,
			 query_results);
      
      // Fix the query node to be the right child, and recurse.
      NWRCdeCommon::BestNodePartners
	(qnode->right(), rnode->left(), rnode->right(), probability, 
	 &rnode_first, &probability_first, &rnode_second, &probability_second);
      bool right_first_result =
	NWRCdeCanonical_(qset, qnode->right(), rnode_first, probability_first,
			 query_results);


      // If the first recursion is carried out exactly, then increment
      // the probability tolerance for the second recursion.
      if(right_first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool right_second_result =
	NWRCdeCanonical_(qset, qnode->right(), rnode_second,
			 probability_second, query_results);

      result = left_first_result && left_second_result &&
	right_first_result && right_second_result;
    }
    
    // Reaccumulate the summary statistics.
    qnode->stat().RefineBoundStatistics(qnode->left()->stat(),
					qnode->right()->stat());
    return result;
  } // end of the case: non-leaf query node.
}
