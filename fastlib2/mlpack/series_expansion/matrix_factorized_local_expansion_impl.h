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
