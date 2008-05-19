#ifndef INSIDE_MATRIX_FACTORIZED_FMM_IMPL_H
#error "This is not a public header file!"
#endif

template<typename TKernelAux>
void MatrixFactorizedFMM<TKernelAux>::BaseCase_
(const Matrix &query_set, const ArrayList<index_t> &query_index_permutation,
 const QueryTree *query_node, const ReferenceTree *reference_node,
 Vector &query_kernel_sums) const {

  
  // Loop over each query point in the query node.
  for(index_t q = query_node->begin(); q < query_node->end(); q++) {
    
    // Get the pointer to the current query point.
    const double *query_point = query_set.GetColumnPtr(q);
    
    // Loop over each reference point in the reference node.
    for(index_t r = reference_node->begin(); r < reference_node->end(); 
	r++) {
      
      // Get the pointer to the current reference point.
      const double *reference_point = reference_set_.GetColumnPtr(r);
      
      // Compute the pairwise distance and kernel value.
      double squared_distance = la::DistanceSqEuclidean(query_set.n_rows(), 
							query_point,
							reference_point);
      double weighted_kernel_value = reference_weights_[r] *
	ka_.kernel_.EvalUnnormOnSq(squared_distance);
      
      query_kernel_sums[query_index_permutation[q]] += weighted_kernel_value;
      
    } // end of iterating over each reference point.
    
  } // end of looping over each query point.
}
