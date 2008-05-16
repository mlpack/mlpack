#ifndef INSIDE_MATRIX_FACTORIZED_FARFIELD_EXPANSION_H
#error "This is not a public header file!"
#endif

#ifndef MATRIX_FACTORIZED_FARFIELD_EXPANSION_IMPL_H
#define MATRIX_FACTORIZED_FARFIELD_EXPANSION_IMPL_H

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::Accumulate
(const Vector &v, double weight, int order) {

  // Implement me sometime!
  DEBUG_ASSERT_MSG(false, "Please implement me!");
}

template<typename TKernelAux>
template<typename Tree>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::AccumulateCoeffs
(const Matrix& reference_set, const Vector& weights, int begin, int end, 
 int order = -1, const Matrix *query_set,
 const ArrayList<Tree *> *query_leaf_nodes = NULL) {
 
  // The sample kernel matrix is S by |R| where |R| is the number of
  // reference points in the reference node and S is the number of
  // query samples taken from the stratification.
  Matrix sample_kernel_matrix;
  int num_reference_samples = (int) sqrt(end - begin);
  int num_query_samples = (int) query_leaf_nodes->size();
  sample_kernel_matrix.Init(num_query_samples, num_reference_samples);
  
  for(index_t r = 0; r < num_reference_samples; r++) {

    // Choose a random reference point.
    index_t random_reference_point_index = math::RandInt(begin, end);
    const double *reference_point =
      reference_set.GetColumnPtr(random_reference_point_index);
    
    for(index_t c = 0; c < num_query_samples; c++) {
      
      // Choose a random query point from the current query strata...
      index_t random_query_point_index =
	math::RandInt(((*query_leaf_nodes)[c])->begin(),
		      ((*query_leaf_nodes)[c])->end());
      const double *query_point =
	query_set->GetColumnPtr(random_query_point_index);	

      // Compute the pairwise distance and the kernel value.
      double squared_distance =
	la::EuclideanSqLength(reference_set.n_rows(), reference_point
			      query_point);
      sample_kernel_matrix.set
	(r, c, (ka_->kernel_).EvalUnnormOnSq(squared_distance));

    } // end of iterating over each sample query strata...
  } // end of iterating over each reference point...
}

template<typename TKernelAux>
double MatrixFactorizedFarFieldExpansion<TKernelAux>::EvaluateField
(const Matrix& data, int row_num, int order) const {

}

template<typename TKernelAux>
double MatrixFactorizedFarFieldExpansion<TKernelAux>::EvaluateField
(const Vector &x_q, int order) const {

}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::Init
(const Vector& center, const TKernelAux &ka) {
  
  // Copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  center_.Copy(center);
  order_ = -1;
  sea_ = &(ka.sea_);
  ka_ = &ka;
}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::Init
(const TKernelAux &ka) {
  
  // Copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);  
  order_ = -1;
  sea_ = &(ka.sea_);
  center_.Init(sea_->get_dimension());
  center_.SetZero();
  ka_ = &ka;
}

template<typename TKernelAux>
template<typename TBound>
int MatrixFactorizedFarFieldExpansion<TKernelAux>::OrderForEvaluating
(const TBound &far_field_region, 
 const TBound &local_field_region, double min_dist_sqd_regions,
 double max_dist_sqd_regions, double max_error, double *actual_error) const {
  
  return ka_->OrderForEvaluatingFarField(far_field_region,
					 local_field_region,
					 min_dist_sqd_regions, 
					 max_dist_sqd_regions, max_error,
					 actual_error);
}

template<typename TKernelAux>
template<typename TBound>
int MatrixFactorizedFarFieldExpansion<TKernelAux>::
OrderForConvertingToLocal(const TBound &far_field_region,
			  const TBound &local_field_region, 
			  double min_dist_sqd_regions, 
			  double max_dist_sqd_regions,
			  double max_error, 
			  double *actual_error) const {
  
  return ka_->OrderForConvertingFromFarFieldToLocal(far_field_region,
						    local_field_region,
						    min_dist_sqd_regions,
						    max_dist_sqd_regions,
						    max_error, actual_error);
}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::PrintDebug
(const char *name, FILE *stream) const {

}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::TranslateFromFarField
(const MatrixFactorizedFarFieldExpansion &se) {

}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::TranslateToLocal
(MatrixFactorizedLocalExpansion<TKernelAux> &se, int truncation_order) {

}

#endif
