#ifndef INSIDE_NWRCDE_H
#error "This is not a public header file!"
#endif

template<typename TKernel>
void NWRCde<TKernel>::NWRCdeBase_(const Matrix &qset, QueryTree *qnode, 
				  ReferenceTree *rnode, double probability, 
				  NWRCdeQueryResult &query_results) {

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().summary.StartReaccumulate();

  // Compute unnormalized sum for each query point.
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {

    // Incorporate the postponed information.
    query_results.ApplyPostponed(qnode->stat().postponed, q);

    // Get the query point.
    const double *q_col = qset.GetColumnPtr(q);
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // pairwise distance and kernel value
      double dsqd = la::DistanceSqEuclidean(qset.n_rows(), q_col, r_col);
      double kernel_value = kernel_.EvalUnnormOnSq(dsqd);
      double weighted_kernel_value = rset_targets_[r] * kernel_value;
      
      query_results.nwr_numerator_sum_l[q] += weighted_kernel_value;
      query_results.nwr_numerator_sum_e[q] += weighted_kernel_value;
      query_results.nwr_denominator_sum_l[q] += kernel_value;
      query_results.nwr_denominator_sum_e[q] += kernel_value;

    } // end of iterating over each reference point.
    
    // Each query point has taken care of all reference points.
    query_results.nwr_numerator_n_pruned[q] += 
      rnode->stat().farfield_expansion_.get_weight_sum();
    query_results.nwr_denominator_n_pruned[q] += ;
    
    // Refine min and max summary statistics.
    qnode->stat().summary.Accumulate(query_results, q);

  } // end of looping over each query point.

  // Clear postponed information.
  qnode->stat().postponed.Reset();
}

template<typename TKernel>
bool NWRCde<TKernel>::NWRCdeCanonical_(const Matrix &qset, QueryTree *qnode,
				       ReferenceTree *rnode, 
				       double probability,
				       NWRCdeResults &query_results) {

  // This is the delta change due to the current query and reference
  // node pair.
  NWRCdeDelta delta;
  delta.Compute(qnode, rnode, kernel_);

  // Try finite difference pruning first.
  if(NWRCdeCommon::ConsiderPairExact(qnode, rnode, probability, delta)) {
    qnode->stat().postponed.ApplyDelta(delta);
    num_finite_difference_prunes_++;    
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
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
      NWRCdeCommon::Heuristic
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
    qnode->left()->stat().postponed.ApplyPostponed(qnode->stat().postponed);
    qnode->right()->stat().postponed.ApplyPostponed(qnode->stat().postponed);
    
    // Clear out the postponed info after being passed down.
    qnode->stat().postponed.Reset();
    
    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      QueryTree *qnode_first = NULL, *qnode_second = NULL;
      double probability_first = 0, probability_second = 0;

      NWRCdeCommon::Heuristic
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
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
  
      // Fix the query node to be the left child, and recurse.
      NWRCdeCommon::Heuristic
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
      NWRCdeCommon::Heuristic
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
    
    // Apply the postponed changes for both child nodes.
    NWRCdeQuerySummary tmp_left_child_summary(qnode->left()->stat().summary);
    tmp_left_child_summary.ApplyPostponed(qnode->left()->stat().postponed);
    NWRCdeQuerySummary tmp_right_child_summary(qnode->right()->stat().summary);
    tmp_right_child_summary.ApplyPostponed(qnode->right()->stat().postponed);

    // Reaccumulate the summary statistics of the current query node.
    qnode->stat().summary.StartReaccumulate();
    qnode->stat().summary.Accumulate(tmp_left_child_summary);
    qnode->stat().summary.Accumulate(tmp_right_child_summary);

    return result;

  } // end of the case: non-leaf query node.
}
