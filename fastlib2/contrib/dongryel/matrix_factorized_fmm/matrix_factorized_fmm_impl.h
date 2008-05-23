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

template<typename TKernelAux>
void MatrixFactorizedFMM<TKernelAux>::CanonicalCase_
(const Matrix &query_set, const ArrayList<index_t> &query_index_permutation,
 QueryTree *query_node, ReferenceTree *reference_node,
 Vector &query_kernel_sums) const {

  // If the current query/reference node is prunable, then
  // approximate.
  double min_distance = sqrt(query_node->bound().MinDistanceSq
			     (reference_node->bound()));
  if(min_distance > std::min(query_node->bound().radius(),
			     reference_node->bound().radius()) &&
     query_node->count() * reference_node->count() >
     query_node->stat().local_expansion_.incoming_skeleton().size() *
     reference_node->stat().farfield_expansion_.outgoing_skeleton().size()) {
    reference_node->stat().farfield_expansion_.TranslateToLocal
      (query_node->stat().local_expansion_, -1, &reference_set_, &query_set);
    return;
  }

  // If the query node is a leaf node,
  if(query_node->is_leaf()) {

    // ... and the reference node is a leaf node, then we do base
    // computation.
    if(reference_node->is_leaf()) {
      BaseCase_(query_set, query_index_permutation, query_node, reference_node,
		query_kernel_sums);
    }
    
    // ...and the reference node is not a leaf node, then recurse on
    // the reference side.
    else {
      CanonicalCase_(query_set, query_index_permutation, query_node,
		     reference_node->left(), query_kernel_sums);
      CanonicalCase_(query_set, query_index_permutation, query_node,
		     reference_node->right(), query_kernel_sums);
    }
  } // end case for the query node as the leaf node.
  
  // If the query node is not a leaf node,
  else {
    
    // ... and the reference node is a leaf node, then recurse on the
    // query side.
    if(reference_node->is_leaf()) {
      CanonicalCase_(query_set, query_index_permutation, query_node->left(),
		     reference_node, query_kernel_sums);
      CanonicalCase_(query_set, query_index_permutation, query_node->right(),
		     reference_node, query_kernel_sums);
    }
    
    // .. and the reference node is not a leaf node, then do the
    // four-way recursion.
    else {
      CanonicalCase_(query_set, query_index_permutation, query_node->left(),
		     reference_node->left(), query_kernel_sums);
      CanonicalCase_(query_set, query_index_permutation, query_node->left(),
		     reference_node->right(), query_kernel_sums);
      CanonicalCase_(query_set, query_index_permutation, query_node->right(),
		     reference_node->left(), query_kernel_sums);
      CanonicalCase_(query_set, query_index_permutation, query_node->right(),
		     reference_node->right(), query_kernel_sums);
    }
  }
}

template<typename TKernelAux>
template<typename Tree>
void MatrixFactorizedFMM<TKernelAux>::GetLeafNodes_
(Tree *node, ArrayList<Tree *> &leaf_nodes) {

  if(node->is_leaf()) {
    leaf_nodes.PushBackCopy(node);
  }
  else {
    GetLeafNodes_(node->left(), leaf_nodes);
    GetLeafNodes_(node->right(), leaf_nodes);
  }
}

template<typename TKernelAux>
void MatrixFactorizedFMM<TKernelAux>::PreProcessQueryTree_
(const Matrix &query_set, QueryTree *query_node, const Matrix &reference_set,
 const ArrayList<ReferenceTree *> &reference_leaf_nodes) {

  // Initialize the local expansion object.
  MatrixFactorizedLocalExpansion<TKernelAux> &local_expansion =
    (query_node->stat()).local_expansion_;
  local_expansion.Init(ka_);

  // For query leaf nodes, train the incoming representation using the
  // set of reference leaf nodes using stratified sampling.
  if(query_node->is_leaf()) {
    local_expansion.TrainBasisFunctions(query_set, query_node->begin(),
					query_node->end(), &reference_set, 
					&reference_leaf_nodes);
  }
  
  // For an internal query node, merge the incoming representations of
  // its children.
  else {
    PreProcessQueryTree_(query_set, query_node->left(), reference_set,
			 reference_leaf_nodes);
    PreProcessQueryTree_(query_set, query_node->right(), reference_set,
			 reference_leaf_nodes);
    
    local_expansion.CombineBasisFunctions
      ((query_node->left()->stat()).local_expansion_,
       (query_node->right()->stat()).local_expansion_);
  }
}

template<typename TKernelAux>
void MatrixFactorizedFMM<TKernelAux>::PreProcessReferenceTree_
(ReferenceTree *reference_node, const Matrix &query_set, 
 const ArrayList<QueryTree *> &query_leaf_nodes) {

  // Initialize the far-field expansion object.
  MatrixFactorizedFarFieldExpansion<TKernelAux> &farfield_expansion =
    (reference_node->stat()).farfield_expansion_;
  farfield_expansion.Init(ka_);

  // For reference leaf nodes, train the outgoing representation using
  // the set of query leaf nodes using stratified sampling.
  if(reference_node->is_leaf()) {
    farfield_expansion.AccumulateCoeffs(reference_set_, reference_weights_,
					reference_node->begin(),
					reference_node->end(),
					-1, &query_set, &query_leaf_nodes);
  }

  // For an internal reference node, merge the representations of its
  // children.
  else {
    PreProcessReferenceTree_(reference_node->left(), query_set,
			     query_leaf_nodes);
    PreProcessReferenceTree_(reference_node->right(), query_set,
			     query_leaf_nodes);

    farfield_expansion.CombineBasisFunctions
      ((reference_node->left()->stat()).farfield_expansion_,
       (reference_node->right()->stat()).farfield_expansion_);
  }
}

template<typename TKernelAux>
void MatrixFactorizedFMM<TKernelAux>::PostProcessQueryTree_
(const Matrix &query_set, const ArrayList<index_t> &query_index_permutation,
 QueryTree *query_node, Vector &query_kernel_sums) const {
  
  const MatrixFactorizedLocalExpansion<TKernelAux> &local_expansion =
    (query_node->stat()).local_expansion_;

  if(query_node->is_leaf()) {
    for(index_t q = query_node->begin(); q < query_node->end(); q++) {
      query_kernel_sums[query_index_permutation[q]] +=
	local_expansion.EvaluateField(query_set, q, query_node->begin());
    }
  }
  else {
    
    local_expansion.TranslateToLocal
      (query_node->left()->stat().local_expansion_);
    local_expansion.TranslateToLocal
      (query_node->right()->stat().local_expansion_);
    PostProcessQueryTree_(query_set, query_index_permutation,
			  query_node->left(), query_kernel_sums);
    PostProcessQueryTree_(query_set, query_index_permutation, 
			  query_node->right(), query_kernel_sums);
  }
}

template<typename TKernelAux>
void MatrixFactorizedFMM<TKernelAux>::Init(const Matrix &references, 
					   struct datanode *module_in) {
  
  // Point to the incoming module.
  module_ = module_in;
  
  // Read in the number of points owned by a leaf
  int leaflen = fx_param_int(module_in, "leaflen", 20);
  
  // Copy reference dataset and reference weights. Currently supports
  // only the uniform weight.
  reference_set_.Copy(references);
  reference_weights_.Init(reference_set_.n_cols());
  reference_weights_.SetAll(1);

  // Construct the reference tree.
  printf("Constructing the reference tree...\n");
  fx_timer_start(fx_root, "reference_tree_construction");
  reference_tree_root_ = proximity::MakeGenMetricTree<ReferenceTree>
    (reference_set_, leaflen, &old_from_new_references_, NULL);

  // Retrieve the list of reference leaf nodes.
  reference_leaf_nodes_.Init();
  GetLeafNodes_(reference_tree_root_, reference_leaf_nodes_);
  fx_timer_stop(fx_root, "reference_tree_construction");
  printf("Finished constructing the reference tree...\n");
  
  // Retrieve the bandwidth and initialize the kernel.
  double bandwidth = fx_param_double_req(module_, "bandwidth");
  ka_.Init(bandwidth, 0, references.n_rows());
}

template<typename TKernelAux>
void MatrixFactorizedFMM<TKernelAux>::Compute
(const Matrix &queries, Vector *query_kernel_sums) {
  
  // Construct the query tree.
  int leaflen = fx_param_int(module_, "leaflen", 20);

  // Copy the query dataset.
  Matrix query_set;
  query_set.Copy(queries);

  printf("Constructing the query tree...\n");
  fx_timer_start(fx_root, "query_tree_construction");
  ArrayList<index_t> old_from_new_queries;
  QueryTree *query_tree_root = 
    proximity::MakeGenMetricTree<QueryTree>
    (query_set, leaflen, &old_from_new_queries, NULL);

  // Retrieve the leaf node lists in the query tree.
  ArrayList<QueryTree *> query_leaf_nodes;
  query_leaf_nodes.Init();
  GetLeafNodes_(query_tree_root, query_leaf_nodes);
  fx_timer_stop(fx_root, "query_tree_construction");
  printf("Finished constructing the query tree...\n");

  printf("Training basis functions...\n");
  fx_timer_start(fx_root, "basis_function_training_time");

  // Train the basis functions in the reference tree and the query
  // tree.
  PreProcessReferenceTree_(reference_tree_root_, query_set, query_leaf_nodes);
  PreProcessQueryTree_(query_set, query_tree_root, reference_set_,
		       reference_leaf_nodes_);

  fx_timer_stop(fx_root, "basis_function_training_time");
  printf("Finished training basis functions...\n");

  // Compute the kernel summations.
  query_kernel_sums->Init(query_set.n_cols());
  query_kernel_sums->SetZero();

  printf("Starting computation...\n");
  fx_timer_start(fx_root, "computation_time");
  CanonicalCase_(query_set, old_from_new_queries, query_tree_root,
		 reference_tree_root_, *query_kernel_sums);


  PostProcessQueryTree_(query_set, old_from_new_queries, query_tree_root, 
			*query_kernel_sums);
  fx_timer_stop(fx_root, "computation_time");
  printf("Computation finished...\n");

  // Delete the query tree after the computation...
  delete query_tree_root;
}
