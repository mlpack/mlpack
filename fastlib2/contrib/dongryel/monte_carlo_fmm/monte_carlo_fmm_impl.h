#ifndef INSIDE_MONTE_CARLO_FMM_H
#error "This is not a public header file!"
#endif

template<typename TKernelAux>
void MonteCarloFMM<TKernelAux>::BaseCase_
(const Matrix &query_set, const ArrayList<index_t> &query_index_permutation,
 QueryTree *query_node, const ReferenceTree *reference_node,
 Vector &query_kernel_sums, double one_sided_probability,
 Vector &query_extra_probability) const {
 
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
      double squared_distance = 
	la::DistanceSqEuclidean(query_set.n_rows(), query_point,
				reference_point);

      double weighted_kernel_value = reference_weights_[r] *
	ka_.kernel_.EvalUnnormOnSq(squared_distance);
      
      query_kernel_sums[query_index_permutation[q]] += weighted_kernel_value;
      
    } // end of iterating over each reference point.

    query_extra_probability[q] += (1 - one_sided_probability);

  } // end of looping over each query point.
}

template<typename TKernelAux>
bool MonteCarloFMM<TKernelAux>::Prunable_
(const QueryTree *query_node, 
 const ArrayList<index_t> &query_index_permutation,
 const ReferenceTree *reference_node, Vector &query_kernel_sums) {
  
  // Try pruning after bound refinement: first compute distance/kernel
  // value bounds.
  DRange dsqd_range, kernel_value_range;
  dsqd_range.lo = query_node->bound().MinDistanceSq(reference_node->bound());
  dsqd_range.hi = query_node->bound().MaxDistanceSq(reference_node->bound());
  kernel_value_range = ka_.kernel_.RangeUnnormOnSq(dsqd_range);

  double used_error = 0.5 * (kernel_value_range.hi - kernel_value_range.lo) *
    reference_node->count();

  // This can probably done more in a smart way...
  double new_mass_l = DBL_MAX;
  for(index_t q = query_node->begin(); q < query_node->end(); q++) {
    new_mass_l = std::min(new_mass_l, 
			  query_kernel_sums[query_index_permutation[q]]);
  }
  new_mass_l += kernel_value_range.lo * reference_node->count();

  double allowed_error = (relative_error_ * new_mass_l) *
    reference_node->count() / ((double) reference_tree_root_->count());

  if(used_error <= allowed_error) {
    for(index_t q = query_node->begin(); q < query_node->end(); q++) {
      query_kernel_sums[query_index_permutation[q]] +=
	kernel_value_range.lo * reference_node->count();
    } 
    return true;
  }
  else {
    return false;
  }
}

template<typename TKernelAux>
bool MonteCarloFMM<TKernelAux>::MonteCarloPrunable_
(const Matrix &query_set, const ArrayList<index_t> &query_index_permutation,
 QueryTree *query_node, const ReferenceTree *reference_node,
 Vector &query_kernel_sums, Vector &query_kernel_sums_scratch_space,
 Vector &query_squared_kernel_sums_scratch_space,
 double one_sided_probability, Vector &query_extra_probability) {

  if(num_initial_samples_per_query_ > reference_node->count()) {
    return false;
  }

  // For each query point in the query node, take samples and
  // determine how many more samples are needed.
  bool flag = true;
  for(index_t q = query_node->begin(); q < query_node->end() && flag; q++) {

    // Compute the required standard score for the given one-sided
    // probability.
    double standard_score = 
      InverseNormalCDF::Compute
      (std::max(one_sided_probability - query_extra_probability[q], 0.0));
    
    // Get the pointer to the current query point.
    const double *query_point = query_set.GetColumnPtr(q);

    // Reset the current position of the scratch space to zero.
    query_kernel_sums_scratch_space[q] = 0;
    query_squared_kernel_sums_scratch_space[q] = 0;
    
    // The initial number of samples is equal to the default.
    int num_samples = num_initial_samples_per_query_;
    int total_samples = 0;

    do {
      for(index_t s = 0; s < num_samples; s++) {
	index_t random_reference_point_index = 
	  math::RandInt(reference_node->begin(), reference_node->end());
	
	// Get the pointer to the current reference point.
	const double *reference_point = 
	  reference_set_.GetColumnPtr(random_reference_point_index);
	
	// Compute the pairwise distance and kernel value.
	double squared_distance = 
	  la::DistanceSqEuclidean(query_set.n_rows(), query_point,
				  reference_point);

	double weighted_kernel_value = 
	  ka_.kernel_.EvalUnnormOnSq(squared_distance);
	query_kernel_sums_scratch_space[q] += weighted_kernel_value;
	query_squared_kernel_sums_scratch_space[q] +=
	  weighted_kernel_value * weighted_kernel_value;
      }
      total_samples += num_samples;

      // Compute the current estimate of the sample mean and the
      // sample variance.
      double sample_mean = query_kernel_sums_scratch_space[q] / 
	((double) total_samples);
      double sample_variance =
	(query_squared_kernel_sums_scratch_space[q] -
	 total_samples * sample_mean * sample_mean) /
	((double) total_samples - 1);

      // Compute the current threshold for guaranteeing the relative
      // error bound.
      int threshold = (sample_variance > DBL_EPSILON) ?
	(int) ceil(math::Sqr(standard_score * reference_tree_root_->count() /
			     relative_error_ / 
			     (sample_mean * reference_node->count() + 
			      query_kernel_sums[query_index_permutation[q]]))
		   * sample_variance):
	0;
      num_samples = threshold - total_samples;

      // If it will require too many samples, give up.
      if(num_samples > reference_node->count()) {
	flag = false;
	break;
      }
      
      // If we are done, then move onto the next query.
      else if(num_samples <= 0) {
	query_kernel_sums_scratch_space[q] /= ((double) total_samples);
	break;
      }

    } while(1);
  } // end of looping over each query...

  // If all queries can be pruned, then add the approximations.
  if(flag) {
    for(index_t q = query_node->begin(); q < query_node->end(); q++) {
      query_kernel_sums[query_index_permutation[q]] += 
	query_kernel_sums_scratch_space[q] * reference_node->count();

      // This can potentially be improved...
      query_extra_probability[q] = 0;
    }    
    return true;
  }
  
  return false;
}

template<typename TKernelAux>
template<typename Tree>
void MonteCarloFMM<TKernelAux>::BestNodePartners_
(Tree *nd, Tree *nd1, Tree *nd2, double probability, 
 Tree **partner1, double *probability1,
 Tree **partner2, double *probability2) {
  
  double d1 = nd->bound().MinDistanceSq(nd1->bound());
  double d2 = nd->bound().MinDistanceSq(nd2->bound());
  
  if(d1 <= d2) {
    *partner1 = nd1;
    *probability1 = sqrt(probability);
    *partner2 = nd2;
    *probability2 = sqrt(probability);
  }
  else {
    *partner1 = nd2;
    *probability1 = sqrt(probability);
    *partner2 = nd1;
    *probability2 = sqrt(probability);
  }
}

template<typename TKernelAux>
void MonteCarloFMM<TKernelAux>::CanonicalCase_
(const Matrix &query_set, const ArrayList<index_t> &query_index_permutation,
 QueryTree *query_node, ReferenceTree *reference_node,
 Vector &query_kernel_sums, Vector &query_kernel_sums_scratch_space,
 Vector &query_squared_kernel_sums_scratch_space,
 double one_sided_probability, Vector &extra_probability) {

  // If prunable, then prune.
  if(MonteCarloPrunable_(query_set, query_index_permutation, query_node, 
			 reference_node, query_kernel_sums, 
			 query_kernel_sums_scratch_space,
			 query_squared_kernel_sums_scratch_space,
			 one_sided_probability, extra_probability)) {
    num_prunes_++;
      return;
  }
  if(Prunable_(query_node, query_index_permutation, reference_node,
	       query_kernel_sums)) {
    num_prunes_++;
    return;
  }

  // If the query node is a leaf node,
  if(query_node->is_leaf()) {

    // ... and the reference node is a leaf node, then we do base
    // computation.
    if(reference_node->is_leaf()) {
      BaseCase_(query_set, query_index_permutation, query_node, reference_node,
		query_kernel_sums, one_sided_probability, extra_probability);
    }
    
    // ...and the reference node is not a leaf node, then recurse on
    // the reference side.
    else {

      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      double one_sided_probability_first = 0;
      double one_sided_probability_second = 0;

      BestNodePartners_(query_node, reference_node->left(),
			reference_node->right(), one_sided_probability,
			&rnode_first, &one_sided_probability_first,
			&rnode_second, &one_sided_probability_second);

      CanonicalCase_(query_set, query_index_permutation, query_node,
		     rnode_first, query_kernel_sums,
		     query_kernel_sums_scratch_space, 
		     query_squared_kernel_sums_scratch_space,
		     one_sided_probability_first, extra_probability);
      CanonicalCase_(query_set, query_index_permutation, query_node,
		     rnode_second, query_kernel_sums,
		     query_kernel_sums_scratch_space, 
		     query_squared_kernel_sums_scratch_space,
		     one_sided_probability_second, extra_probability);
    }
  } // end case for the query node as the leaf node.
  
  // If the query node is not a leaf node,
  else {
    
    // ... and the reference node is a leaf node, then recurse on the
    // query side.
    if(reference_node->is_leaf()) {
      CanonicalCase_(query_set, query_index_permutation, query_node->left(),
		     reference_node, query_kernel_sums, 
		     query_kernel_sums_scratch_space,
		     query_squared_kernel_sums_scratch_space,
		     one_sided_probability, extra_probability);
      CanonicalCase_(query_set, query_index_permutation, query_node->right(),
		     reference_node, query_kernel_sums, 
		     query_kernel_sums_scratch_space,
		     query_squared_kernel_sums_scratch_space,
		     one_sided_probability, extra_probability);
    }
    
    // .. and the reference node is not a leaf node, then do the
    // four-way recursion.
    else {

      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      double one_sided_probability_first = 0;
      double one_sided_probability_second = 0;

      BestNodePartners_(query_node->left(), reference_node->left(),
			reference_node->right(), one_sided_probability,
			&rnode_first, &one_sided_probability_first,
			&rnode_second, &one_sided_probability_second);

      CanonicalCase_(query_set, query_index_permutation, query_node->left(),
		     rnode_first, query_kernel_sums, 
		     query_kernel_sums_scratch_space, 
		     query_squared_kernel_sums_scratch_space,
		     one_sided_probability_first, extra_probability);
      CanonicalCase_(query_set, query_index_permutation, query_node->left(),
		     rnode_second, query_kernel_sums, 
		     query_kernel_sums_scratch_space, 
		     query_squared_kernel_sums_scratch_space,
		     one_sided_probability_second, extra_probability);

      BestNodePartners_(query_node->right(), reference_node->left(),
			reference_node->right(), one_sided_probability,
			&rnode_first, &one_sided_probability_first,
			&rnode_second, &one_sided_probability_second);

      CanonicalCase_(query_set, query_index_permutation, query_node->right(),
		     rnode_first, query_kernel_sums, 
		     query_kernel_sums_scratch_space, 
		     query_squared_kernel_sums_scratch_space,
		     one_sided_probability_first, extra_probability);
      CanonicalCase_(query_set, query_index_permutation, query_node->right(),
		     rnode_second, query_kernel_sums, 
		     query_kernel_sums_scratch_space, 
		     query_squared_kernel_sums_scratch_space,
		     one_sided_probability_second, extra_probability);
    }
  }
}

template<typename TKernelAux>
void MonteCarloFMM<TKernelAux>::Init(const Matrix &references, 
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
  fx_timer_stop(fx_root, "reference_tree_construction");
  printf("Finished constructing the reference tree...\n");
  
  // Retrieve the bandwidth and initialize the kernel.
  double bandwidth = fx_param_double_req(module_, "bandwidth");
  ka_.Init(bandwidth, 0, references.n_rows());

  // Retrieve the required relative error level.
  relative_error_ = fx_param_double(module_, "relative_error", 0.01);
}

template<typename TKernelAux>
void MonteCarloFMM<TKernelAux>::Compute
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

  fx_timer_stop(fx_root, "query_tree_construction");
  printf("Finished constructing the query tree...\n");

  // Compute the kernel summations.
  query_kernel_sums->Init(query_set.n_cols());
  query_kernel_sums->SetZero();

  // Initialize the scratch space for Monte Carlo sampling.
  Vector query_kernel_sums_scratch_space;
  Vector query_squared_kernel_sums_scratch_space;
  query_kernel_sums_scratch_space.Init(query_set.n_cols());
  query_kernel_sums_scratch_space.SetZero();
  query_squared_kernel_sums_scratch_space.Init(query_set.n_cols());
  query_squared_kernel_sums_scratch_space.SetZero();

  printf("Starting computation...\n");
  fx_timer_start(fx_root, "computation_time");
  num_prunes_ = 0;
  
  // Get the probability for guaranteeing the results.
  double probability = fx_param_double(module_, "probability", 0.75);
  double one_sided_probability = probability + 0.5 * (1 - probability);
  Vector query_extra_probability;
  query_extra_probability.Init(query_set.n_cols());
  query_extra_probability.SetZero();

  CanonicalCase_(query_set, old_from_new_queries, query_tree_root,
		 reference_tree_root_, *query_kernel_sums, 
		 query_kernel_sums_scratch_space,
		 query_squared_kernel_sums_scratch_space,
		 one_sided_probability, query_extra_probability);
  fx_timer_stop(fx_root, "computation_time");
  printf("Computation finished...\n");
  printf("Number of prunes: %d\n", num_prunes_);

  // Delete the query tree after the computation...
  delete query_tree_root;
}
