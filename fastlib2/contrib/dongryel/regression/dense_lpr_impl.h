// Make sure this file is included only in dense_lpr.h. This is not a
// public header file!
#ifndef INSIDE_DENSE_LPR_H
#error "This file is not a public header file!"
#endif

#include "mlpack/series_expansion/bounds_aux.h"
#include "matrix_util.h"

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::BasicComputeDualTree_
(const Matrix &queries, Vector *query_regression_estimates,
 ArrayList<DRange> *query_confidence_bands,
 Vector *query_magnitude_weight_diagrams, Vector *query_influence_values) {
  
  // Set the relative error tolerance.
  relative_error_ = fx_param_double(module_, "relative_error", 0.01);
  internal_relative_error_ = relative_error_ / (relative_error_ + 2.0);
  
  // Copy the query set.
  Matrix qset;
  qset.Copy(queries);
  
  // read in the number of points owned by a leaf
  int leaflen = fx_param_int(module_, "leaflen", 20);
  
  // Construct the query tree.
  ArrayList<index_t> old_from_new_queries;
  QueryTree *qroot = tree::MakeKdTreeMidpoint<QueryTree>
    (qset, leaflen, &old_from_new_queries, NULL);
      
  // Initialize storage space for intermediate computations.
  Matrix numerator_l, numerator_e;
  Vector numerator_used_error, numerator_n_pruned;
  ArrayList<Matrix> denominator_l, denominator_e;
  Vector denominator_used_error, denominator_n_pruned;
  numerator_l.Init(row_length_, queries.n_cols());
  numerator_e.Init(row_length_, queries.n_cols());
  numerator_used_error.Init(queries.n_cols());
  numerator_n_pruned.Init(queries.n_cols());
  denominator_l.Init(queries.n_cols());
  denominator_e.Init(queries.n_cols());
  for(index_t i = 0; i < queries.n_cols(); i++) {
    denominator_l[i].Init(row_length_, row_length_);
    denominator_e[i].Init(row_length_, row_length_);
  }
  denominator_used_error.Init(queries.n_cols());
  denominator_n_pruned.Init(queries.n_cols());      
  ArrayList<Matrix> weight_diagram_numerator_l, weight_diagram_numerator_e;
  Vector weight_diagram_used_error;
  weight_diagram_numerator_l.Init(queries.n_cols());
  weight_diagram_numerator_e.Init(queries.n_cols());
  for(index_t i = 0; i < queries.n_cols(); i++) {
    weight_diagram_numerator_l[i].Init(row_length_, row_length_);
    weight_diagram_numerator_e[i].Init(row_length_, row_length_);
  }
  weight_diagram_used_error.Init(queries.n_cols());
  
  // Initialize storage for the final results.
  query_regression_estimates->Init(queries.n_cols());
  query_magnitude_weight_diagrams->Init(queries.n_cols());
  query_influence_values->Init(queries.n_cols());
  
  // Three steps: initialize the query tree, then call dualtree,
  // then final postprocess.
  InitializeQueryTree_(qroot, numerator_l, numerator_e,
		       numerator_used_error, numerator_n_pruned,
		       denominator_l, denominator_e, 
		       denominator_used_error, denominator_n_pruned,
		       weight_diagram_numerator_l,
		       weight_diagram_numerator_e,
		       weight_diagram_used_error);
  DualtreeLprCanonical_
    (qroot, rroot_, qset, numerator_l, numerator_e, numerator_used_error, 
     numerator_n_pruned, denominator_l, denominator_e,
     denominator_used_error, denominator_n_pruned,
     weight_diagram_numerator_l, weight_diagram_numerator_e,
     weight_diagram_used_error);
  FinalizeQueryTree_
    (qroot, qset, query_regression_estimates, 
     query_magnitude_weight_diagrams, query_influence_values,
     numerator_l, numerator_e, numerator_used_error, numerator_n_pruned, 
     denominator_l, denominator_e, denominator_used_error, 
     denominator_n_pruned, weight_diagram_numerator_l, 
     weight_diagram_numerator_e, weight_diagram_used_error);
  
  // After the computation, we do not need the query tree, so we
  // free it.
  delete qroot;
  
  // Reshuffle the results to account for dataset reshuffling
  // resulted from tree constructions
  Vector tmp_q_results;
  tmp_q_results.Init(query_regression_estimates->length());
  
  for(index_t i = 0; i < tmp_q_results.length(); i++) {
    tmp_q_results[old_from_new_queries[i]] = 
      (*query_regression_estimates)[i];
      }
  for(index_t i = 0; i < tmp_q_results.length(); i++) {
    (*query_regression_estimates)[i] = tmp_q_results[i];
  }
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::BasicComputeSingleTree_
(const Matrix &queries, Vector *query_regression_estimates,
 ArrayList<DRange> *query_confidence_bands,
 Vector *query_magnitude_weight_diagrams, Vector *query_influence_values) {
        
  // Set the relative error tolerance.
  relative_error_ = fx_param_double(module_, "relative_error", 0.01);
  internal_relative_error_ = relative_error_ / (relative_error_ + 2.0);
  
  // read in the number of points owned by a leaf
  int leaflen = fx_param_int(module_, "leaflen", 20);
  
  // Initialize storage space for intermediate computations.
  Matrix numerator_l, numerator_e;
  Vector numerator_used_error, numerator_n_pruned;
  ArrayList<Matrix> denominator_l, denominator_e;
  Vector denominator_used_error, denominator_n_pruned;
  numerator_l.Init(row_length_, 1);
  numerator_e.Init(row_length_, 1);
  numerator_used_error.Init(1);
  numerator_n_pruned.Init(1);
  denominator_l.Init(1);
  denominator_e.Init(1);
  for(index_t i = 0; i < 1; i++) {
    denominator_l[i].Init(row_length_, row_length_);
    denominator_e[i].Init(row_length_, row_length_);
  }
  denominator_used_error.Init(1);
  denominator_n_pruned.Init(1);      
  ArrayList<Matrix> weight_diagram_numerator_l, weight_diagram_numerator_e;
  Vector weight_diagram_used_error;
  weight_diagram_numerator_l.Init(1);
  weight_diagram_numerator_e.Init(1);
  for(index_t i = 0; i < 1; i++) {
    weight_diagram_numerator_l[i].Init(row_length_, row_length_);
    weight_diagram_numerator_e[i].Init(row_length_, row_length_);
  }
  weight_diagram_used_error.Init(1);

  // Initialize storage for the final results.
  query_regression_estimates->Init(queries.n_cols());
  query_magnitude_weight_diagrams->Init(queries.n_cols());
  query_influence_values->Init(queries.n_cols());

  // iterate over each query point.
  for(index_t q = 0; q < queries.n_cols(); q++) {

    // Make each column query vector as the whole dataset.
    Vector q_col;
    queries.MakeColumnVector(q, &q_col);
    Vector q_col_copy;
    q_col_copy.Copy(q_col);
    Matrix qset;
    qset.AliasColVector(q_col_copy);

    // Make an appropriate alias of the final storage.
    Vector query_regression_estimates_alias, query_influence_values_alias;
    Vector query_magnitude_weight_diagrams_alias;
    query_regression_estimates_alias.Alias
      ((query_regression_estimates->ptr()) + q, 1);
    query_magnitude_weight_diagrams_alias.Alias
      ((query_magnitude_weight_diagrams->ptr()) + q, 1);
    query_influence_values_alias.Alias((query_influence_values->ptr()) + q, 1);

    // Construct the query tree.
    QueryTree *qroot = tree::MakeKdTreeMidpoint<QueryTree>
      (qset, leaflen, NULL, NULL);
    
    // Three steps: initialize the query tree, then call dualtree,
    // then final postprocess.
    InitializeQueryTree_(qroot, numerator_l, numerator_e,
			 numerator_used_error, numerator_n_pruned,
			 denominator_l, denominator_e, 
			 denominator_used_error, denominator_n_pruned,
			 weight_diagram_numerator_l, 
			 weight_diagram_numerator_e,
			 weight_diagram_used_error);
    DualtreeLprCanonical_
      (qroot, rroot_, qset, numerator_l, numerator_e, numerator_used_error, 
       numerator_n_pruned, denominator_l, denominator_e,
       denominator_used_error, denominator_n_pruned,
       weight_diagram_numerator_l, weight_diagram_numerator_e,
       weight_diagram_used_error);
    FinalizeQueryTree_
      (qroot, qset, &query_regression_estimates_alias,
       &query_magnitude_weight_diagrams_alias, &query_influence_values_alias,
       numerator_l, numerator_e, numerator_used_error, numerator_n_pruned, 
       denominator_l, denominator_e, denominator_used_error,
       denominator_n_pruned, weight_diagram_numerator_l,
       weight_diagram_numerator_e, weight_diagram_used_error);
    
    // After the computation, we do not need the query tree, so we
    // free it.
    delete qroot;

  } // end of iterating over each query.
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::SqdistAndKernelRanges_
(QueryTree *qnode, ReferenceTree *rnode, DRange &dsqd_range, 
 DRange &kernel_value_range) {

  // The following assumes that you are using a monotonically
  // decreasing kernel!
  dsqd_range = qnode->bound().RangeDistanceSq(rnode->bound());
  kernel_value_range.lo =
    rnode->stat().min_bandwidth_kernel.EvalUnnormOnSq(dsqd_range.hi);
  kernel_value_range.hi =
    rnode->stat().max_bandwidth_kernel.EvalUnnormOnSq(dsqd_range.lo);
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::ResetQuery_
(int q, Matrix &numerator_l, Matrix &numerator_e, Vector &numerator_used_error,
 Vector &numerator_n_pruned, ArrayList<Matrix> &denominator_l,
 ArrayList<Matrix> &denominator_e, Vector &denominator_used_error,
 Vector &denominator_n_pruned, ArrayList<Matrix> &weight_diagram_numerator_l,
 ArrayList<Matrix> &weight_diagram_numerator_e,
 Vector &weight_diagram_used_error) {
  
  // First the numerator quantities.
  Vector q_numerator_l, q_numerator_e;
  numerator_l.MakeColumnVector(q, &q_numerator_l);
  numerator_e.MakeColumnVector(q, &q_numerator_e);
  q_numerator_l.SetZero();
  q_numerator_e.SetZero();
  numerator_used_error[q] = 0;
  numerator_n_pruned[q] = 0;
  
  // Then the denominator quantities,
  denominator_l[q].SetZero();
  denominator_e[q].SetZero();
  denominator_used_error[q] = 0;
  denominator_n_pruned[q] = 0;

  // The quantities related to the weight diagram computation
  // (confidence band).
  weight_diagram_numerator_l[q].SetZero();
  weight_diagram_numerator_e[q].SetZero();
  weight_diagram_used_error[q] = 0;
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::
InitializeReferenceStatistics_(ReferenceTree *rnode) {
  
  if(rnode->is_leaf()) {
    
    // Temporary vector for computing the reference point expansion.
    Vector reference_point_expansion;
    reference_point_expansion.Init(row_length_);

    // Clear the sum statistics before accumulating.
    (rnode->stat().sum_target_weighted_data_).SetZero();

    // For a leaf reference node, iterate over each reference point
    // and compute the weighted vector and tally these up for the
    // sum statistics owned by the reference node.
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the pointer to the current reference point.
      Vector r_col;
      rset_.MakeColumnVector(r, &r_col);

      // Get the pointer to the reference column to be updated.
      double *r_target_weighted_by_coordinates = 
	target_weighted_rset_.GetColumnPtr(r);

      // Compute the multiindex expansion of the given reference point.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, r_col.ptr(), reference_point_expansion.ptr());
      
      // Scale the expansion by the reference target.
      la::ScaleOverwrite
	(row_length_, rset_targets_[r], reference_point_expansion.ptr(),
	 r_target_weighted_by_coordinates);
      
      // Accumulate the far field coefficient for the target weighted
      // reference vector and the outerproduct. The outer loop
      // iterates over each column and the inner iterates over each
      // row.
      for(index_t j = 0; j < row_length_; j++) {
	rnode->stat().target_weighted_data_far_field_expansion_[j].
	  Add(r_target_weighted_by_coordinates[j], kernels_[r].bandwidth_sq(),
	      r_col);

	for(index_t i = 0; i < row_length_; i++) {
	  rnode->stat().data_outer_products_far_field_expansion_[j][i].
	    Add(reference_point_expansion[j] * reference_point_expansion[i],
		kernels_[r].bandwidth_sq(), r_col);
	}
      }

      // Tally up the weighted targets.
      la::AddTo(row_length_, r_target_weighted_by_coordinates,
		(rnode->stat().sum_target_weighted_data_).ptr());
      
      // Accumulate the bandwidth statistics.
      rnode->stat().min_bandwidth_kernel.Init
	(std::min(sqrt(rnode->stat().min_bandwidth_kernel.bandwidth_sq()),
		  sqrt(kernels_[r].bandwidth_sq())));
      rnode->stat().max_bandwidth_kernel.Init
	(std::max(sqrt(rnode->stat().max_bandwidth_kernel.bandwidth_sq()),
		  sqrt(kernels_[r].bandwidth_sq())));

    } // end of iterating over each reference point.
    
    // Compute Frobenius norm of the accumulated sum
    rnode->stat().sum_target_weighted_data_error_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);
    rnode->stat().sum_target_weighted_data_alloc_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);
  }  
  else {
    
    // Recursively call the function with left and right and merge.
    InitializeReferenceStatistics_(rnode->left());
    InitializeReferenceStatistics_(rnode->right());
   
    // Compute the sum of the sub sums.
    la::AddOverwrite((rnode->left()->stat()).sum_target_weighted_data_,
		     (rnode->right()->stat()).sum_target_weighted_data_,
		     &(rnode->stat().sum_target_weighted_data_));
    rnode->stat().sum_target_weighted_data_error_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);
    rnode->stat().sum_target_weighted_data_alloc_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);

    // Translate far-field moments of the child to form the parent.
    for(index_t j = 0; j < row_length_; j++) {
      rnode->stat().target_weighted_data_far_field_expansion_[j].
	Add(rnode->left()->stat().
	    target_weighted_data_far_field_expansion_[j]);
      rnode->stat().target_weighted_data_far_field_expansion_[j].
	Add(rnode->right()->stat().
	    target_weighted_data_far_field_expansion_[j]);
      
      for(index_t i = 0; i < row_length_; i++) {

	// First the far field moments of outer product using the bandwidth
	rnode->stat().data_outer_products_far_field_expansion_[j][i].
	  Add(rnode->left()->stat().
	      data_outer_products_far_field_expansion_[j][i]);
	rnode->stat().data_outer_products_far_field_expansion_[j][i].
	  Add(rnode->right()->stat().
	      data_outer_products_far_field_expansion_[j][i]);
      }  // end of iterating over each row.
    } // end of iterating over each column.
    
    rnode->stat().min_bandwidth_kernel.Init
      (std::min
       (sqrt(rnode->left()->stat().min_bandwidth_kernel.bandwidth_sq()),
	sqrt(rnode->right()->stat().min_bandwidth_kernel.bandwidth_sq())));
    rnode->stat().max_bandwidth_kernel.Init
      (std::max
       (sqrt(rnode->left()->stat().max_bandwidth_kernel.bandwidth_sq()),
	sqrt(rnode->right()->stat().max_bandwidth_kernel.bandwidth_sq())));
					   
  } // end of the non-leaf case.
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::
InitializeQueryTree_(QueryTree *qnode, Matrix &numerator_l, 
		     Matrix &numerator_e, Vector &numerator_used_error,
		     Vector &numerator_n_pruned, 
		     ArrayList<Matrix> &denominator_l,
		     ArrayList<Matrix> &denominator_e,
		     Vector &denominator_used_error,
		     Vector &denominator_n_pruned,
		     ArrayList<Matrix> &weight_diagram_numerator_l,
		     ArrayList<Matrix> &weight_diagram_numerator_e,
		     Vector &weight_diagram_used_error) {
    
  // Set the bounds to default values for the statistics.
  qnode->stat().SetZero();

  // If the query node is a leaf, then initialize the corresponding
  // bound quantities for each query point.
  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Reset the bounds corresponding to the particular query point.
      ResetQuery_(q, numerator_l, numerator_e, numerator_used_error,
		  numerator_n_pruned, denominator_l, denominator_e,
		  denominator_used_error, denominator_n_pruned,
		  weight_diagram_numerator_l, weight_diagram_numerator_e,
		  weight_diagram_used_error);
    }
  }

  // Otherwise, then traverse to the left and the right.
  else {
    InitializeQueryTree_(qnode->left(), numerator_l, numerator_e,
			 numerator_used_error, numerator_n_pruned,
			 denominator_l, denominator_e, denominator_used_error,
			 denominator_n_pruned, weight_diagram_numerator_l,
			 weight_diagram_numerator_e, 
			 weight_diagram_used_error);
    InitializeQueryTree_(qnode->right(), numerator_l, numerator_e,
			 numerator_used_error, numerator_n_pruned,
			 denominator_l, denominator_e, denominator_used_error,
			 denominator_n_pruned, weight_diagram_numerator_l,
			 weight_diagram_numerator_e, 
			 weight_diagram_used_error);
  }
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::BestNodePartners_
(QueryTree *nd, ReferenceTree *nd1, ReferenceTree *nd2, 
 ReferenceTree **partner1, ReferenceTree **partner2) {
  
  double d1 = nd->bound().MinDistanceSq(nd1->bound());
  double d2 = nd->bound().MinDistanceSq(nd2->bound());
  
  if(d1 <= d2) {
    *partner1 = nd1;
    *partner2 = nd2;
  }
  else {
    *partner1 = nd2;
    *partner2 = nd1;
  }
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::BestNodePartners_
(ReferenceTree *nd, QueryTree *nd1, QueryTree *nd2, 
 QueryTree **partner1, QueryTree **partner2) {
  
  double d1 = nd->bound().MinDistanceSq(nd1->bound());
  double d2 = nd->bound().MinDistanceSq(nd2->bound());
  
  if(d1 <= d2) {
    *partner1 = nd1;
    *partner2 = nd2;
  }
  else {
    *partner1 = nd2;
    *partner2 = nd1;
  }
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::DualtreeLprBase_
(QueryTree *qnode, ReferenceTree *rnode, const Matrix &qset,
 Matrix &numerator_l, 
 Matrix &numerator_e, Vector &numerator_used_error, 
 Vector &numerator_n_pruned, ArrayList<Matrix> &denominator_l, 
 ArrayList<Matrix> &denominator_e, Vector &denominator_used_error, 
 Vector &denominator_n_pruned, ArrayList<Matrix> &weight_diagram_numerator_l,
 ArrayList<Matrix> &weight_diagram_numerator_e,
 Vector &weight_diagram_numerator_used_error) {

  // Temporary variable for storing multivariate expansion of a
  // reference point.
  Vector reference_point_expansion;
  reference_point_expansion.Init(row_length_);

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().numerator_norm_l_ = DBL_MAX;
  qnode->stat().numerator_used_error_ = 0;
  qnode->stat().numerator_n_pruned_ = DBL_MAX;
  qnode->stat().denominator_norm_l_ = DBL_MAX;
  qnode->stat().kernel_sum_l_ = DBL_MAX;
  qnode->stat().denominator_used_error_ = 0;
  qnode->stat().denominator_n_pruned_ = DBL_MAX;
  qnode->stat().weight_diagram_numerator_norm_l_ = DBL_MAX;
  qnode->stat().weight_diagram_numerator_used_error_ = 0;
  
  // Iterate over each query point.
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // Get the query point.
    const double *q_col = qset.GetColumnPtr(q);

    // Get the query-relevant quantities to be updated.
    double *q_numerator_l = numerator_l.GetColumnPtr(q);
    double *q_numerator_e = numerator_e.GetColumnPtr(q);

    // Incorporate the postponed information for the numerator vector.
    la::AddTo(row_length_, qnode->stat().postponed_numerator_l_.ptr(),
	      q_numerator_l);
    numerator_used_error[q] += qnode->stat().postponed_numerator_used_error_;
    numerator_n_pruned[q] += qnode->stat().postponed_numerator_n_pruned_;

    // Incorporate the postponed information for the denominator matrix.
    la::AddTo(qnode->stat().postponed_denominator_l_, &(denominator_l[q]));
    denominator_used_error[q] += 
      qnode->stat().postponed_denominator_used_error_;
    denominator_n_pruned[q] += qnode->stat().postponed_denominator_n_pruned_;

    // Incorporate the postponed information for the weight diagram
    // numerator matrix.
    la::AddTo(qnode->stat().postponed_weight_diagram_numerator_l_, 
	      &(weight_diagram_numerator_l[q]));
    weight_diagram_numerator_used_error[q] += 
      qnode->stat().postponed_weight_diagram_numerator_used_error_;

    // Iterate over each reference point.
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the reference point and its training value.
      const double *r_col = rset_.GetColumnPtr(r);

      // Compute the reference point expansion.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, r_col, reference_point_expansion.ptr());

      // Pairwise distance and kernel value and kernel value weighted
      // by the reference target training value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernels_[r].EvalUnnormOnSq(dsqd);
      double target_weighted_kernel_value = rset_targets_[r] * kernel_value;
      
      // Loop over each column of the matrix to be updated.
      for(index_t j = 0; j < row_length_; j++) {

	// Tally the sum up for the numerator vector B^T W(q) Y.
	q_numerator_l[j] += target_weighted_kernel_value * 
	  reference_point_expansion[j];
	q_numerator_e[j] += target_weighted_kernel_value *
	  reference_point_expansion[j];
	
	// Loop over each row of the matrix to be updated.
	for(index_t i = 0; i < row_length_; i++) {
	  
	  // Tally the sum up for the denominator matrix B^T W(q) B.
	  denominator_l[q].set(i, j, denominator_l[q].get(i, j) +
				kernel_value * reference_point_expansion[i] *
				reference_point_expansion[j]);
	  denominator_e[q].set(i, j, denominator_e[q].get(i, j) +
				kernel_value * reference_point_expansion[i] *
				reference_point_expansion[j]);

	  // Tally up the sum for the weight diagram numerator matrix
	  // B^T W(q) W(q) B.
	  weight_diagram_numerator_l[q].set
	    (i, j, weight_diagram_numerator_l[q].get(i, j) +
	     kernel_value * kernel_value * reference_point_expansion[i] *
	     reference_point_expansion[j]);
	  weight_diagram_numerator_e[q].set
	    (i, j, weight_diagram_numerator_e[q].get(i, j) +
	     kernel_value * kernel_value * reference_point_expansion[i] *
	     reference_point_expansion[j]);	  
	  
	} // End of iterating over each row.
      } // End of iterating over each column.

    } // End of iterating over each reference point.
    
    // Each query point has taken care of all reference points.
    numerator_n_pruned[q] += 
      rnode->stat().sum_target_weighted_data_alloc_norm_;
    denominator_n_pruned[q] +=
      rnode->stat().sum_data_outer_products_alloc_norm_;
    
    // Refine min summary statistics for the numerator.
    qnode->stat().numerator_norm_l_ =
      std::min(qnode->stat().numerator_norm_l_,
	       MatrixUtil::EntrywiseLpNorm(row_length_, q_numerator_l, 1));
    qnode->stat().numerator_used_error_ =
      std::max(qnode->stat().numerator_used_error_, numerator_used_error[q]);
    qnode->stat().numerator_n_pruned_ =
      std::min(qnode->stat().numerator_n_pruned_, numerator_n_pruned[q]);
    
    // Refine summary statistics for the denominator.
    qnode->stat().denominator_norm_l_ =
      std::min(qnode->stat().denominator_norm_l_,
	       MatrixUtil::EntrywiseLpNorm(denominator_l[q], 1));
    qnode->stat().kernel_sum_l_ =
      std::min(qnode->stat().kernel_sum_l_, denominator_l[q].get(0, 0));
    qnode->stat().denominator_used_error_ =
      std::max(qnode->stat().denominator_used_error_, 
	       denominator_used_error[q]);
    qnode->stat().denominator_n_pruned_ =
      std::min(qnode->stat().denominator_n_pruned_, denominator_n_pruned[q]);

    // Refine summary statistics for the weight diagram numerator.
    qnode->stat().weight_diagram_numerator_norm_l_ =
      std::min(qnode->stat().weight_diagram_numerator_norm_l_,
	       MatrixUtil::EntrywiseLpNorm(weight_diagram_numerator_l[q], 1));
    qnode->stat().weight_diagram_numerator_used_error_ =
      std::max(qnode->stat().weight_diagram_numerator_used_error_, 
	       weight_diagram_numerator_used_error[q]);

  } // End of iterating over each query point.
  
  // Clear postponed information for the numerator matrix.
  qnode->stat().postponed_numerator_l_.SetZero();
  qnode->stat().postponed_numerator_used_error_ = 0;
  qnode->stat().postponed_numerator_n_pruned_ = 0;

  // Clear postponed information for the denominator matrix.
  qnode->stat().postponed_denominator_l_.SetZero();
  qnode->stat().postponed_denominator_used_error_ = 0;
  qnode->stat().postponed_denominator_n_pruned_ = 0;  

  // Clear postponed information for the weight diagram numerator
  // matrix.
  qnode->stat().postponed_weight_diagram_numerator_l_.SetZero();
  qnode->stat().postponed_weight_diagram_numerator_used_error_ = 0;
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::DualtreeLprCanonical_
(QueryTree *qnode, ReferenceTree *rnode, const Matrix &qset,
 Matrix &numerator_l, Matrix &numerator_e, Vector &numerator_used_error, 
 Vector &numerator_n_pruned, ArrayList<Matrix> &denominator_l, 
 ArrayList<Matrix> &denominator_e, Vector &denominator_used_error, 
 Vector &denominator_n_pruned, ArrayList<Matrix> &weight_diagram_numerator_l,
 ArrayList<Matrix> &weight_diagram_numerator_e, 
 Vector &weight_diagram_used_error) {

  // Total amount of used error
  double delta_numerator_used_error, delta_denominator_used_error,
    delta_weight_diagram_numerator_used_error;
  
  // Total portion accounted by pruning.
  double delta_numerator_n_pruned, delta_denominator_n_pruned;

  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range, kernel_value_range;
  
  // Temporary variable for holding lower and estimate changes.
  Vector numerator_dl, numerator_de;
  numerator_dl.Init(row_length_);
  numerator_de.Init(row_length_);
  Matrix denominator_dl, denominator_de, weight_diagram_numerator_dl,
    weight_diagram_numerator_de;
  denominator_dl.Init(row_length_, row_length_);
  denominator_de.Init(row_length_, row_length_);
  weight_diagram_numerator_dl.Init(row_length_, row_length_);
  weight_diagram_numerator_de.Init(row_length_, row_length_);
  
  // Compute distance ranges and kernel ranges first.
  SqdistAndKernelRanges_(qnode, rnode, dsqd_range, kernel_value_range);

  // Try finite difference pruning first
  if(TPruneRule::Prunable
     (internal_relative_error_, 
      rroot_->stat().sum_target_weighted_data_alloc_norm_,
      rroot_->stat().sum_data_outer_products_alloc_norm_,
      qnode, rnode, dsqd_range, kernel_value_range,
      numerator_dl, numerator_de, delta_numerator_used_error, 
      delta_numerator_n_pruned, denominator_dl, denominator_de, 
      delta_denominator_used_error, delta_denominator_n_pruned,
      weight_diagram_numerator_dl, weight_diagram_numerator_de,
      delta_weight_diagram_numerator_used_error)) {
    
    la::AddTo(numerator_dl, &(qnode->stat().postponed_numerator_l_));
    la::AddTo(numerator_de, &(qnode->stat().postponed_numerator_e_));
    qnode->stat().postponed_numerator_used_error_ += 
      delta_numerator_used_error;
    qnode->stat().postponed_numerator_n_pruned_ += delta_numerator_n_pruned;
    
    la::AddTo(denominator_dl, &(qnode->stat().postponed_denominator_l_));
    la::AddTo(denominator_de, &(qnode->stat().postponed_denominator_e_));
    qnode->stat().postponed_denominator_used_error_ += 
      delta_denominator_used_error;
    qnode->stat().postponed_denominator_n_pruned_ += 
      delta_denominator_n_pruned;

    la::AddTo(weight_diagram_numerator_dl, 
	      &(qnode->stat().postponed_weight_diagram_numerator_l_));
    la::AddTo(weight_diagram_numerator_de, 
	      &(qnode->stat().postponed_weight_diagram_numerator_e_));
    qnode->stat().postponed_weight_diagram_numerator_used_error_ += 
      delta_weight_diagram_numerator_used_error;

    // Keep track of the number of finite difference prunes.
    num_finite_difference_prunes_++;
    return;
  }

  // For the Epanechnikov kernel, we can prune using the far field
  // moments if the maximum distance between the two nodes is within
  // the bandwidth! This if-statement does not apply to the Gaussian
  // kernel, so I need to fix in the future!
  if(rnode->stat().min_bandwidth_kernel.bandwidth_sq() >= dsqd_range.hi && 
     rnode->count() > dimension_ * dimension_) {

    for(index_t j = 0; j < row_length_; j++) {
      
      qnode->stat().postponed_numerator_l_[j] += 
	rnode->stat().target_weighted_data_far_field_expansion_[j].
	ComputeMinKernelSum(qnode->bound());
      qnode->stat().postponed_moment_numerator_e_[j].
	Add(rnode->stat().target_weighted_data_far_field_expansion_[j]);

      for(index_t i = 0; i < row_length_; i++) {

	qnode->stat().postponed_denominator_l_.set
	  (j, i, qnode->stat().postponed_denominator_l_.get(j, i) +
	   rnode->stat().data_outer_products_far_field_expansion_[j][i].
	   ComputeMinKernelSum(qnode->bound()));
	qnode->stat().postponed_moment_denominator_e_[j][i].
	  Add(rnode->stat().data_outer_products_far_field_expansion_[j][i]);
	
	qnode->stat().postponed_weight_diagram_numerator_l_.set
	  (j, i, qnode->stat().postponed_weight_diagram_numerator_l_.get(j, i) 
	   + rnode->stat().
	   data_outer_products_far_field_expansion_[j][i].
	   ComputeMinKernelSum(qnode->bound()));
	qnode->stat().postponed_moment_weight_diagram_numerator_e_[j][i].
	  Add(rnode->stat().data_outer_products_far_field_expansion_[j][i]);
      }
    }

    qnode->stat().postponed_numerator_n_pruned_ += delta_numerator_n_pruned;
    qnode->stat().postponed_denominator_n_pruned_ += 
      delta_denominator_n_pruned;

    // Keep track of the far-field prunes.
    num_far_field_prunes_++;
    return;
  }

  // for leaf query node
  if(qnode->is_leaf()) {

    // for leaf pairs, go exhaustive
    if(rnode->is_leaf()) {
      DualtreeLprBase_(qnode, rnode, qset, numerator_l, numerator_e, 
		       numerator_used_error, numerator_n_pruned,
		       denominator_l, denominator_e, denominator_used_error, 
		       denominator_n_pruned, weight_diagram_numerator_l,
		       weight_diagram_numerator_e, weight_diagram_used_error);
      return;
    }

    // for non-leaf reference, expand reference node
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      BestNodePartners_(qnode, rnode->left(), rnode->right(), &rnode_first,
                        &rnode_second);
      DualtreeLprCanonical_
	(qnode, rnode_first, qset, numerator_l, numerator_e, 
	 numerator_used_error, numerator_n_pruned, denominator_l, 
	 denominator_e, denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e, 
	 weight_diagram_used_error);
      DualtreeLprCanonical_
	(qnode, rnode_second, qset, numerator_l, numerator_e, 
	 numerator_used_error, numerator_n_pruned, denominator_l, 
	 denominator_e, denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e, 
	 weight_diagram_used_error);
      return;
    }
  }
  
  // for non-leaf query node
  else {
    
    LprQStat &q_stat = qnode->stat();
    LprQStat &q_left_stat = qnode->left()->stat();
    LprQStat &q_right_stat = qnode->right()->stat();

    // Push down postponed bound changes owned by the current query
    // node to the children of the query node.
    la::AddTo(q_stat.postponed_numerator_l_,
	      &q_left_stat.postponed_numerator_l_);
    la::AddTo(q_stat.postponed_numerator_l_,
	      &q_right_stat.postponed_numerator_l_);
    q_left_stat.postponed_numerator_used_error_ += 
      q_stat.postponed_numerator_used_error_;
    q_right_stat.postponed_numerator_used_error_ += 
      q_stat.postponed_numerator_used_error_;
    q_left_stat.postponed_numerator_n_pruned_ += 
      q_stat.postponed_numerator_n_pruned_;
    q_right_stat.postponed_numerator_n_pruned_ += 
      q_stat.postponed_numerator_n_pruned_;
    
    la::AddTo(q_stat.postponed_denominator_l_,
	      &q_left_stat.postponed_denominator_l_);
    la::AddTo(q_stat.postponed_denominator_l_,
	      &q_right_stat.postponed_denominator_l_);
    q_left_stat.postponed_denominator_used_error_ += 
      q_stat.postponed_denominator_used_error_;
    q_right_stat.postponed_denominator_used_error_ += 
      q_stat.postponed_denominator_used_error_;
    q_left_stat.postponed_denominator_n_pruned_ += 
      q_stat.postponed_denominator_n_pruned_;
    q_right_stat.postponed_denominator_n_pruned_ += 
      q_stat.postponed_denominator_n_pruned_;

    la::AddTo(q_stat.postponed_weight_diagram_numerator_l_,
	      &q_left_stat.postponed_weight_diagram_numerator_l_);
    la::AddTo(q_stat.postponed_weight_diagram_numerator_l_,
	      &q_right_stat.postponed_weight_diagram_numerator_l_);
    q_left_stat.postponed_weight_diagram_numerator_used_error_ += 
      q_stat.postponed_weight_diagram_numerator_used_error_;
    q_right_stat.postponed_weight_diagram_numerator_used_error_ += 
      q_stat.postponed_weight_diagram_numerator_used_error_;

    // Clear the passed down postponed information.
    q_stat.postponed_numerator_l_.SetZero();
    q_stat.postponed_numerator_used_error_ = 0;
    q_stat.postponed_numerator_n_pruned_ = 0;
    q_stat.postponed_denominator_l_.SetZero();
    q_stat.postponed_denominator_used_error_ = 0;
    q_stat.postponed_denominator_n_pruned_ = 0;
    q_stat.postponed_weight_diagram_numerator_l_.SetZero();
    q_stat.postponed_weight_diagram_numerator_used_error_ = 0;
    
    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      QueryTree *qnode_first = NULL, *qnode_second = NULL;
      
      BestNodePartners_(rnode, qnode->left(), qnode->right(), &qnode_first,
			&qnode_second);
      DualtreeLprCanonical_
	(qnode_first, rnode, qset, numerator_l, numerator_e, 
	 numerator_used_error, numerator_n_pruned, denominator_l, 
	 denominator_e, denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e, 
	 weight_diagram_used_error);
      DualtreeLprCanonical_
	(qnode_second, rnode, qset, numerator_l, numerator_e, 
	 numerator_used_error, numerator_n_pruned, denominator_l, 
	 denominator_e, denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e, 
	 weight_diagram_used_error);
    }
    
    // for non-leaf reference node, expand both query and reference nodes
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      
      BestNodePartners_(qnode->left(), rnode->left(), rnode->right(),
			&rnode_first, &rnode_second);
      DualtreeLprCanonical_
	(qnode->left(), rnode_first, qset, numerator_l, numerator_e, 
	 numerator_used_error, numerator_n_pruned, denominator_l, 
	 denominator_e, denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e, 
	 weight_diagram_used_error);
      DualtreeLprCanonical_
	(qnode->left(), rnode_second, qset, numerator_l, numerator_e, 
	 numerator_used_error, numerator_n_pruned, denominator_l, 
	 denominator_e, denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e, 
	 weight_diagram_used_error);
      
      BestNodePartners_(qnode->right(), rnode->left(), rnode->right(),
			&rnode_first, &rnode_second);
      DualtreeLprCanonical_
	(qnode->right(), rnode_first, qset, numerator_l, numerator_e, 
	 numerator_used_error, numerator_n_pruned, denominator_l, 
	 denominator_e, denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e, 
	 weight_diagram_used_error);
      DualtreeLprCanonical_
	(qnode->right(), rnode_second, qset, numerator_l, numerator_e, 
	 numerator_used_error, numerator_n_pruned, denominator_l, 
	 denominator_e, denominator_used_error, denominator_n_pruned,
	 weight_diagram_numerator_l, weight_diagram_numerator_e, 
	 weight_diagram_used_error);
    }
    
    // reaccumulate the summary statistics.
    q_stat.numerator_norm_l_ = 
      std::min
      (q_left_stat.numerator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_left_stat.postponed_numerator_l_, 1),
       q_right_stat.numerator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_right_stat.postponed_numerator_l_, 1));
    q_stat.numerator_used_error_ = 
      std::max(q_left_stat.numerator_used_error_,
	       q_right_stat.numerator_used_error_);
    q_stat.numerator_n_pruned_ = 
      std::min(q_left_stat.numerator_n_pruned_,
	       q_right_stat.numerator_n_pruned_);
    q_stat.denominator_norm_l_ =
      std::min
      (q_left_stat.denominator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_left_stat.postponed_denominator_l_, 1),
       q_right_stat.denominator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_right_stat.postponed_denominator_l_, 1));
    q_stat.kernel_sum_l_ =
      std::min(q_left_stat.kernel_sum_l_ +
	       q_left_stat.postponed_denominator_l_.get(0, 0),
	       q_right_stat.kernel_sum_l_ +
	       q_right_stat.postponed_denominator_l_.get(0, 0));
    q_stat.denominator_used_error_ = 
      std::max(q_left_stat.denominator_used_error_,
	       q_right_stat.denominator_used_error_);
    q_stat.denominator_n_pruned_ = 
      std::min(q_left_stat.denominator_n_pruned_,
	       q_right_stat.denominator_n_pruned_);    

    q_stat.weight_diagram_numerator_norm_l_ =
      std::min
      (q_left_stat.weight_diagram_numerator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm
       (q_left_stat.postponed_weight_diagram_numerator_l_, 1),
       q_right_stat.weight_diagram_numerator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm
       (q_right_stat.postponed_weight_diagram_numerator_l_, 1));
    q_stat.weight_diagram_numerator_used_error_ = 
      std::max(q_left_stat.weight_diagram_numerator_used_error_,
	       q_right_stat.weight_diagram_numerator_used_error_);
    return;
  } // end of the case: non-leaf query node.  
}

template<typename TKernel, typename TPruneRule>
void DenseLpr<TKernel, TPruneRule>::
FinalizeQueryTree_(QueryTree *qnode, const Matrix &qset,
		   Vector *query_regression_estimates,
		   Vector *query_magnitude_weight_diagrams,
		   Vector *query_influence_values,
		   Matrix &numerator_l, Matrix &numerator_e, 
		   Vector &numerator_used_error, Vector &numerator_n_pruned,
		   ArrayList<Matrix> &denominator_l, 
		   ArrayList<Matrix> &denominator_e,
		   Vector &denominator_used_error, 
		   Vector &denominator_n_pruned,
		   ArrayList<Matrix> &weight_diagram_numerator_l,
		   ArrayList<Matrix> &weight_diagram_numerator_e,
		   Vector &weight_diagram_used_error) {
  
  LprQStat &q_stat = qnode->stat();

  if(qnode->is_leaf()) {

    Matrix pseudoinverse_denominator;
    pseudoinverse_denominator.Init(row_length_, row_length_);
    Vector least_squares_solution;
    least_squares_solution.Init(row_length_);
    Vector query_point_expansion;
    query_point_expansion.Init(row_length_);

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Get the query point.
      Vector q_col;
      qset.MakeColumnVector(q, &q_col);
      const double *query_point = q_col.ptr();

      // Get the numerator vectors accumulating the sums to update.
      Vector q_numerator_l, q_numerator_e;
      numerator_l.MakeColumnVector(q, &q_numerator_l);
      numerator_e.MakeColumnVector(q, &q_numerator_e);

      // Incorporate the postponed information for the numerator.
      la::AddTo(q_stat.postponed_numerator_l_, &q_numerator_l);
      la::AddTo(q_stat.postponed_numerator_e_, &q_numerator_e);

      // Incorporate the postponed information for the denominator.
      la::AddTo(q_stat.postponed_denominator_l_, &(denominator_l[q]));
      la::AddTo(q_stat.postponed_denominator_e_, &(denominator_e[q]));

      // Incorporate the postponed information for the weight diagram.
      la::AddTo(q_stat.postponed_weight_diagram_numerator_l_,
		&(weight_diagram_numerator_l[q]));
      la::AddTo(q_stat.postponed_weight_diagram_numerator_e_,
		&(weight_diagram_numerator_e[q]));

      // Incorporate the postponed estimates using the Epanechnikov
      // series expansion.
      for(index_t i = 0; i < row_length_; i++) {
	q_numerator_e[i] += 
	  qnode->stat().postponed_moment_numerator_e_[i].
	  ComputeKernelSum(q_col);
	for(index_t j = 0; j < row_length_; j++) {
	  denominator_e[q].set
	    (j, i, denominator_e[q].get(j, i) +
	     qnode->stat().postponed_moment_denominator_e_[j][i].
	     ComputeKernelSum(q_col));
	  weight_diagram_numerator_e[q].set
	    (j, i, weight_diagram_numerator_e[q].get(j, i) +
	     qnode->stat().postponed_moment_weight_diagram_numerator_e_[j][i].
	     ComputeSquaredKernelSum(q_col));
	}
      }

      // After incorporating all of the postponed information,
      // finalize the regression estimate by solving the appropriate
      // linear system (B^T W(q) B) z(q) = B^T W(q) Y for z(q) and
      // taking the dot product between z(q) and the polynomial power
      // formed from the query point coordinates.
      MatrixUtil::PseudoInverse(denominator_e[q], &pseudoinverse_denominator);

      la::MulOverwrite(pseudoinverse_denominator, q_numerator_e,
		       &least_squares_solution);

      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, query_point, query_point_expansion.ptr());
      (*query_regression_estimates)[q] = la::Dot(query_point_expansion,
						 least_squares_solution);
      
      // Now we compute the magnitude of the weight diagram for each
      // query point.
      Vector pseudo_inverse_times_query_expansion;
      Vector intermediate_product;
      la::MulInit(pseudoinverse_denominator, query_point_expansion,
		  &pseudo_inverse_times_query_expansion);
      la::MulInit(weight_diagram_numerator_e[q],
		  pseudo_inverse_times_query_expansion, &intermediate_product);
      (*query_magnitude_weight_diagrams)[q] =
	sqrt(la::Dot(pseudo_inverse_times_query_expansion, 
		     intermediate_product));

      // Compute the influence value at each point (if it belongs to
      // the reference set), i.e. (r(q))^T (B^T W(q) B)^-1 B^T W(q)
      // e_i = (r(q))^T (B^T W(q) B)-1 r(q).
      if(query_influence_values != NULL) {
	(*query_influence_values)[q] =
	  la::Dot(query_point_expansion, pseudo_inverse_times_query_expansion);
      }
    }
  }
  else {
    
    LprQStat &q_left_stat = qnode->left()->stat();
    LprQStat &q_right_stat = qnode->right()->stat();

    // Push down approximations for the numerator.
    la::AddTo(q_stat.postponed_numerator_l_,
	      &(q_left_stat.postponed_numerator_l_));
    la::AddTo(q_stat.postponed_numerator_e_,
	      &(q_left_stat.postponed_numerator_e_));
    la::AddTo(q_stat.postponed_numerator_l_,
              &(q_right_stat.postponed_numerator_l_));
    la::AddTo(q_stat.postponed_numerator_e_,
              &(q_right_stat.postponed_numerator_e_));
    
    // Push down approximations for the denominator.
    la::AddTo(q_stat.postponed_denominator_l_,
              &(q_left_stat.postponed_denominator_l_));
    la::AddTo(q_stat.postponed_denominator_e_,
              &(q_left_stat.postponed_denominator_e_));
    la::AddTo(q_stat.postponed_denominator_l_,
              &(q_right_stat.postponed_denominator_l_));
    la::AddTo(q_stat.postponed_denominator_e_,
              &(q_right_stat.postponed_denominator_e_));

    // Push down approximations for the weight diagram.
    la::AddTo(q_stat.postponed_weight_diagram_numerator_l_,
	      &(q_left_stat.postponed_weight_diagram_numerator_l_));
    la::AddTo(q_stat.postponed_weight_diagram_numerator_e_,
	      &(q_left_stat.postponed_weight_diagram_numerator_e_));
    la::AddTo(q_stat.postponed_weight_diagram_numerator_l_,
	      &(q_right_stat.postponed_weight_diagram_numerator_l_));   
    la::AddTo(q_stat.postponed_weight_diagram_numerator_e_, 
	      &(q_right_stat.postponed_weight_diagram_numerator_e_));

    // Push down Epanechnikov series expansion pruning.
    for(index_t i = 0; i < row_length_; i++) {
      q_left_stat.postponed_moment_numerator_e_[i].Add
	(q_stat.postponed_moment_numerator_e_[i]);
      q_right_stat.postponed_moment_numerator_e_[i].Add
	(q_stat.postponed_moment_numerator_e_[i]);

      for(index_t j = 0; j < row_length_; j++) {
	q_left_stat.postponed_moment_denominator_e_[i][j].Add
	  (q_stat.postponed_moment_denominator_e_[i][j]);
	q_right_stat.postponed_moment_denominator_e_[i][j].Add
	  (q_stat.postponed_moment_denominator_e_[i][j]);
	q_left_stat.postponed_moment_weight_diagram_numerator_e_[i][j].Add
	  (q_stat.postponed_moment_weight_diagram_numerator_e_[i][j]);
	q_right_stat.postponed_moment_weight_diagram_numerator_e_[i][j].Add
	  (q_stat.postponed_moment_weight_diagram_numerator_e_[i][j]);
      }
    }

    FinalizeQueryTree_(qnode->left(), qset, query_regression_estimates,
		       query_magnitude_weight_diagrams, query_influence_values,
		       numerator_l, numerator_e, numerator_used_error, 
		       numerator_n_pruned, denominator_l, denominator_e, 
		       denominator_used_error, denominator_n_pruned,
		       weight_diagram_numerator_l, weight_diagram_numerator_e, 
		       weight_diagram_used_error);
    FinalizeQueryTree_(qnode->right(), qset, query_regression_estimates,
		       query_magnitude_weight_diagrams, query_influence_values,
		       numerator_l, numerator_e, numerator_used_error, 
		       numerator_n_pruned, denominator_l, denominator_e, 
		       denominator_used_error, denominator_n_pruned,
		       weight_diagram_numerator_l, weight_diagram_numerator_e,
		       weight_diagram_used_error);
  }
}
