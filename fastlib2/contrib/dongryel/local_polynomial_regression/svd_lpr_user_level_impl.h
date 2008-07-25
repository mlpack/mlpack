#ifndef INSIDE_SVD_LPR_H
#error "This is not a public header file!"
#endif

template<typename TKernel>
void SvdLpr<TKernel>::Init(Matrix &reference_set, Matrix &reference_targets,
			   struct datanode *module_in) {
  
  // Set the incoming parameter module.
  module_ = module_in;

  // Read in the number of points owned by a leaf.
  int leaflen = fx_param_int(module_in, "leaflen", 40);
  
  // Set the dimension, local polynomial approximation order and the
  // number of coefficients.
  lpr_order_ = fx_param_int_req(module_in, "lpr_order");
  num_lpr_coeffs_ = (int) math::BinomialCoefficient(dimension_ + lpr_order_,
						    dimension_);
  
  // Copy matrices.
  reference_set_.Copy(reference_set);
  reference_targets_.Copy(reference_targets);

  // Set the z-score necessary for computing the confidence band.
  z_score_ = fx_param_double(module_, "z_score", 1.96);

  // Train the model (compute mean squared error and confidence
  // interval at each training point).
  

}

template<typename TKernel>
void SvdLpr<TKernel>::Compute(const Matrix &queries, 
			      Vector *query_regression_estimates,
			      ArrayList<DRange> *query_confidence_bands,
			      Vector *query_magnitude_weight_diagrams) {
  
  
}

