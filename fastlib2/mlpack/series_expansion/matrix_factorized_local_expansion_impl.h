#ifndef INSIDE_MATRIX_FACTORIZED_LOCAL_EXPANSION_H
#error "This is not a public header file!"
#endif

#ifndef MATRIX_FACTORIZED_LOCAL_EXPANSION_IMPL_H
#define MATRIX_FACTORIZED_LOCAL_EXPANSION_IMPL_H

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::PrintDebug
(const char *name, FILE *stream) const {

}

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::CombineBasisFunctions
(MatrixFactorizedLocalExpansion &local_expansion1,
 MatrixFactorizedLocalExpansion &local_expansion2) {
  
  // The incoming skeleton for an internal node is formed by
  // concatenating the incoming skeleton of its children.
  const ArrayList<index_t> &incoming_skeleton1 =
    local_expansion1.incoming_skeleton();
  const ArrayList<index_t> &incoming_skeleton2 =
    local_expansion2.incoming_skeleton();

  incoming_skeleton_.Init(incoming_skeleton1.size() + 
			  incoming_skeleton2.size());
  for(index_t i = 0; i < incoming_skeleton1.size(); i++) {
    incoming_skeleton_[i] = incoming_skeleton1[i];
  }
  for(index_t i = incoming_skeleton1.size(); i < incoming_skeleton_.size();
      i++) {
    incoming_skeleton_[i] = incoming_skeleton2[i - incoming_skeleton1.size()];
  }

  // Allocate space for local moments based on the size of the
  // incoming skeleton.
  coeffs_.Init(incoming_skeleton_.size());
  coeffs_.SetZero();

  // Compute the beginning index and the count of the local expansion
  // for the children expansions.
  local_expansion1.set_local_to_local_translation_begin(0);
  local_expansion1.set_local_to_local_translation_count
    (incoming_skeleton1.size());
  local_expansion2.set_local_to_local_translation_begin
    (incoming_skeleton1.size());
  local_expansion2.set_local_to_local_translation_count
    (incoming_skeleton2.size());
}

template<typename TKernelAux>
double MatrixFactorizedLocalExpansion<TKernelAux>::EvaluateField
(const Matrix& data, int row_num, int begin_row_num) const {
  
  // Take the dot product of the (row_num - begin_row_num) th row of
  // the evaluation operator.
  double dot_product = 0;

  for(index_t i = 0; i < evaluation_operator_->n_cols(); i++) {
    dot_product += 
      evaluation_operator_->get(row_num - begin_row_num, i) * coeffs_[i];
  }
  return dot_product;
}

template<typename TKernelAux>
double MatrixFactorizedLocalExpansion<TKernelAux>::EvaluateField
(const Vector& x_q) const {
  DEBUG_ASSERT_MSG(false, "Please implement me!");
  return -1;
}

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::Init
(const Vector& center, const TKernelAux &ka) {
  
  // Copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  ka_ = &ka;

  // Set the incoming representation to be null. This is only valid
  // for a leaf node.
  evaluation_operator_ = NULL;
}

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::Init(const TKernelAux &ka) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  ka_ = &ka;

  // Set the incoming representation to be null. This is only valid
  // for a leaf node.
  evaluation_operator_ = NULL;
}

template<typename TKernelAux>
template<typename TBound>
int MatrixFactorizedLocalExpansion<TKernelAux>::OrderForEvaluating
(const TBound &far_field_region, 
 const TBound &local_field_region, double min_dist_sqd_regions,
 double max_dist_sqd_regions, double max_error, double *actual_error) const {
  
  return ka_->OrderForEvaluatingLocal(far_field_region, local_field_region, 
				     min_dist_sqd_regions,
				     max_dist_sqd_regions, max_error, 
				     actual_error);
}

template<typename TKernelAux>
template<typename Tree>
void MatrixFactorizedLocalExpansion<TKernelAux>::TrainBasisFunctions
(const Matrix &query_set, int begin, int end, const Matrix *reference_set,
 const ArrayList<Tree *> *reference_leaf_nodes) {
  
  // The sample kernel matrix is |Q| by S where |Q| is the number of
  // query points in the query node and S is the number of reference
  // samples taken from the stratification.
  Matrix sample_kernel_matrix_transposed;
  int num_reference_samples = reference_leaf_nodes->size();
  int num_query_samples = end - begin;

  // Allocate a temporary space for holding the indices of the query
  // points, from which the incoming skeleton will be chosen.
  ArrayList<index_t> tmp_incoming_skeleton;
  tmp_incoming_skeleton.Init(num_query_samples);
  for(index_t q = 0; q < num_query_samples; q++) {

    // Choose a random query point and record its index.
    tmp_incoming_skeleton[q] = q + begin;
  }
  // Sort the chosen query indices and eliminate duplicates...
  qsort(tmp_incoming_skeleton.begin(), tmp_incoming_skeleton.size(),
	sizeof(index_t), &qsort_compar_);
  remove_duplicates_in_sorted_array_(tmp_incoming_skeleton);
  num_query_samples = tmp_incoming_skeleton.size();

  // After determining the number of query samples to take,
  // allocate the space for the sample kernel matrix to be computed.
  sample_kernel_matrix_transposed.Init(num_reference_samples, 
				       num_query_samples);
  
  for(index_t r = 0; r < num_reference_samples; r++) {

    // Choose a random reference point from the current reference strata...
    index_t random_reference_point_index =
      math::RandInt(((*reference_leaf_nodes)[r])->begin(),
		    ((*reference_leaf_nodes)[r])->end());
    const double *reference_point =
      reference_set->GetColumnPtr(random_reference_point_index);

    for(index_t c = 0; c < num_query_samples; c++) {
      
      // The current query point
      const double *query_point = 
	query_set.GetColumnPtr(tmp_incoming_skeleton[c]);	

      // Compute the pairwise distance and the kernel value.
      double squared_distance =
	la::DistanceSqEuclidean(query_set.n_rows(), reference_point,
				query_point);
      double kernel_value = (ka_->kernel_).EvalUnnormOnSq(squared_distance);
      sample_kernel_matrix_transposed.set
	(r, c, ((*reference_leaf_nodes)[r])->count() * kernel_value);
      
    } // end of iterating over each sample query strata...
  } // end of iterating over each reference point...
  
  // CUR-decompose the sample kernel matrix.
  ArrayList<index_t> row_indices;
  Matrix evaluation_operator_transposed;
  CURDecomposition::ExactCompute(sample_kernel_matrix_transposed, 
				 &evaluation_operator_transposed,
				 &row_indices);
  
  // The incoming skeleton is constructed from the sampled rows in the
  // matrix factorization.
  incoming_skeleton_.Init(row_indices.size());
  for(index_t s = 0; s < row_indices.size(); s++) {
    incoming_skeleton_[s] = tmp_incoming_skeleton[row_indices[s]];
  }

  // Compute the evaluation operator, which is the transpose of the
  // evaluation operator computed from the decomposition.
  la::TransposeInit(evaluation_operator_transposed, evaluation_operator_);

  // Allocate space based on the size of the incoming skeleton.
  coeffs_.Init(incoming_skeleton_.size());
  coeffs_.SetZero();
}

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::TranslateToLocal
(MatrixFactorizedLocalExpansion &se) const {
  
  // Local-to-local translation involves determining the indices of
  // the query points that belong to the local moment to be
  // translated.
  index_t beginning_index = se.local_to_local_translation_begin();
  index_t count = se.local_to_local_translation_count();
  
  // Reference to the destination coefficients.
  Vector &destination_coeffs = se.coeffs();

  for(index_t i = 0; i < count; i++) {
    destination_coeffs[i] += coeffs_[i + beginning_index];
  }
}

#endif
