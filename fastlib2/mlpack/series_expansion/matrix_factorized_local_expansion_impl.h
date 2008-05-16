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
void LocalExpansion<TKernelAux>::Init(const Vector& center,
				      const TKernelAux &ka) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  center_.Copy(center);
  order_ = -1;
  sea_ = &(ka.sea_);
  ka_ = &ka;
}

template<typename TKernelAux>
void LocalExpansion<TKernelAux>::Init(const TKernelAux &ka) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  order_ = -1;
  sea_ = &(ka.sea_);
  center_.Init(sea_->get_dimension());
  ka_ = &ka;
}

template<typename TKernelAux>
template<typename TBound>
int LocalExpansion<TKernelAux>::OrderForEvaluating
(const TBound &far_field_region, 
 const TBound &local_field_region, double min_dist_sqd_regions,
 double max_dist_sqd_regions, double max_error, double *actual_error) const {
  
  return ka_->OrderForEvaluatingLocal(far_field_region, local_field_region, 
				     min_dist_sqd_regions,
				     max_dist_sqd_regions, max_error, 
				     actual_error);
}

template<typename TKernelAux>
void LocalExpansion<TKernelAux>::TranslateFromFarField
(const FarFieldExpansion<TKernelAux> &se) {

}
  
template<typename TKernelAux>
void LocalExpansion<TKernelAux>::TranslateToLocal(LocalExpansion &se) {

}

#endif
