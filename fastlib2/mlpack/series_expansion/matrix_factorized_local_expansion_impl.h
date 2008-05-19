#ifndef INSIDE_MATRIX_FACTORIZED_LOCAL_EXPANSION_H
#error "This is not a public header file!"
#endif

#ifndef MATRIX_FACTORIZED_LOCAL_EXPANSION_IMPL_H
#define MATRIX_FACTORIZED_LOCAL_EXPANSION_IMPL_H


template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::AccumulateCoeffs
(const Matrix& data, const Vector& weights, int begin, int end, int order) {

}

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::PrintDebug
(const char *name, FILE *stream) const {

}

template<typename TKernelAux>
double MatrixFactorizedLocalExpansion<TKernelAux>::EvaluateField
(const Matrix& data, int row_num) const {

}

template<typename TKernelAux>
double MatrixFactorizedLocalExpansion<TKernelAux>::EvaluateField
(const Vector& x_q) const {

}

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::Init
(const Vector& center, const TKernelAux &ka) {
  
  // Copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  center_.Copy(center);
  order_ = -1;
  sea_ = &(ka.sea_);
  ka_ = &ka;

  // Set the incoming representation to be null. This is only valid
  // for a leaf node.
  incoming_representation_ = NULL;
}

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::Init(const TKernelAux &ka) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  order_ = -1;
  sea_ = &(ka.sea_);
  center_.Init(sea_->get_dimension());
  ka_ = &ka;

  // Set the incoming representation to be null. This is only valid
  // for a leaf node.
  incoming_representation_ = NULL;
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
  Matrix sample_kernel_matrix;
  int num_reference_samples = reference_leaf_nodes->size();
  int num_query_samples = (int) sqrt(end - begin);

  // Allocate a temporary space for holding the indices of the query
  // points, from which the incoming skeleton will be chosen.
  ArrayList<index_t> tmp_incoming_skeleton;
  tmp_incoming_skeleton.Init(num_query_samples);
  for(index_t q = 0; q < num_query_samples; q++) {

    // Choose a random query point and record its index.
    index_t random_query_point_index = math::RandInt(begin, end);
    tmp_incoming_skeleton[r] = random_query_point_index;
  }
  // Sort the chosen query indices and eliminate duplicates...
  qsort(tmp_incoming_skeleton.begin(), tmp_incoming_skeleton.size(),
	sizeof(index_t), &qsort_compar_);
  remove_duplicates_in_sorted_array_(tmp_incoming_skeleton);
  num_query_samples = tmp_incoming_skeleton.size();

  // After determining the number of query samples to take,
  // allocate the space for the sample kernel matrix to be computed.
  sample_kernel_matrix.Init(num_query_samples, num_reference_samples);
  
  for(index_t r = 0; r < num_reference_samples; r++) {

    // Choose a random reference point from the current reference strata...
    index_t random_reference_point_index =
      math::RandInt(((*reference_leaf_nodes)[r])->begin(),
		    ((*reference_leaf_nodes)[r])->end());
    const double *reference_point =
      reference_set.GetColumnPtr(random_reference_point_index);

    for(index_t c = 0; c < num_query_samples; c++) {
      
      // The current query point
      const double *query_point = 
	query_set->GetColumnPtr(tmp_incoming_skeleton[c]);	

      // Compute the pairwise distance and the kernel value.
      double squared_distance =
	la::DistanceSqEuclidean(reference_set.n_rows(), reference_point,
				query_point);
      sample_kernel_matrix.set
	(c, r, (ka_->kernel_).EvalUnnormOnSq(squared_distance));
      
    } // end of iterating over each sample query strata...
  } // end of iterating over each reference point...
  
  // CUR-decompose the sample kernel matrix.
  Matrix c_mat, u_mat, r_mat;
  ArrayList<index_t> column_indices, row_indices;
  CURDecomposition::Compute(sample_kernel_matrix, &c_mat, &u_mat, &r_mat,
			    &column_indices, &row_indices);
  
  // The incoming skeleton is constructed from the sampled rows in the
  // matrix factorization.
  incoming_skeleton_ = new ArrayList<index_t>();
  incoming_skeleton_->Init(row_indices.size());
  for(index_t s = 0; s < row_indices.size(); s++) {
    (*incoming_skeleton_)[s] = tmp_incoming_skeleton[row_incides[s]];
  }

  // Compute the evaluation operator, which is the product of the C
  // and the U factor appropriately scaled by the row scaled R factor.
  la::MulInit(c_mat, u_mat, evaluation_operator_);
  for(index_t i = 0; i < r_mat.n_rows(); i++) {
    double scaling_factor =
      (sample_kernel_matrix.get(row_indices[i], 0) < DBL_EPSILON) ?
      0:r_mat.get(i, 0) / sample_kernel_matrix.get(row_indices[i], 0);

    la::Scale(evaluation_operator_->n_rows(), scaling_factor,
	      evaluation_operator_->GetColumnPtr(i));
  }  
}

template<typename TKernelAux>
void MatrixFactorizedLocalExpansion<TKernelAux>::TranslateToLocal
(MatrixFactorizedLocalExpansion &se) {
  
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
