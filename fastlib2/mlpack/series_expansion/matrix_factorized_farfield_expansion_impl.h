#ifndef INSIDE_MATRIX_FACTORIZED_FARFIELD_EXPANSION_H
#error "This is not a public header file!"
#endif

#ifndef MATRIX_FACTORIZED_FARFIELD_EXPANSION_IMPL_H
#define MATRIX_FACTORIZED_FARFIELD_EXPANSION_IMPL_H

#include "fastlib/fastlib.h"
#include "cur_decomposition.h"

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
 int order, const Matrix *query_set, 
 const ArrayList<Tree *> *query_leaf_nodes) {

  // The sample kernel matrix is S by |R| where |R| is the number of
  // reference points in the reference node and S is the number of
  // query samples taken from the stratification.
  Matrix sample_kernel_matrix;
  int num_reference_samples = end - begin;
  int num_query_samples = query_leaf_nodes->size();

  ArrayList<index_t> tmp_outgoing_skeleton;
  tmp_outgoing_skeleton.Init(num_reference_samples);

  // The temporary outgoing skeleton includes all of the reference
  // points.
  for(index_t r = 0; r < num_reference_samples; r++) {
    tmp_outgoing_skeleton[r] = begin + r;
  }
  num_reference_samples = tmp_outgoing_skeleton.size();

  // After determining the number of reference samples to take,
  // allocate the space for the sample kernel matrix to be computed.
  sample_kernel_matrix.Init(num_query_samples, num_reference_samples);

  for(index_t r = 0; r < num_reference_samples; r++) {

    // The reference point...
    const double *reference_point =
      reference_set.GetColumnPtr(tmp_outgoing_skeleton[r]);

    for(index_t c = 0; c < query_leaf_nodes->size(); c++) {
      
      // Choose a random query point from the current query strata...
      index_t random_query_point_index =
	math::RandInt(((*query_leaf_nodes)[c])->begin(),
		      ((*query_leaf_nodes)[c])->end());

      const double *query_point =
	query_set->GetColumnPtr(random_query_point_index);	

      // Compute the pairwise distance and the kernel value.
      double squared_distance =
	la::DistanceSqEuclidean
	(reference_set.n_rows(), reference_point, query_point);
      double kernel_value = (ka_->kernel_).EvalUnnormOnSq(squared_distance);
      sample_kernel_matrix.set(c, r, kernel_value *
			       (((*query_leaf_nodes)[c])->count()));
      
    } // end of iterating over each sample query strata...
  } // end of iterating over each reference point...

  // CUR-decompose the sample kernel matrix.
  Matrix c_mat, u_mat, r_mat;
  ArrayList<index_t> column_indices, row_indices;
  CURDecomposition::Compute(sample_kernel_matrix,
			    &c_mat, &u_mat, &r_mat,
			    &column_indices, &row_indices);
  
  // Compute the reconstruction error by multiplying the reference
  // weights by the kernel matrix approximation difference, reweighted
  // by the inverse of the query points contained in each query
  // strata.
  Matrix reconstruction_error, intermediate_matrix;
  la::MulInit(c_mat, u_mat, &intermediate_matrix);
  reconstruction_error.Copy(sample_kernel_matrix);
  la::MulExpert(-1.0, intermediate_matrix, r_mat, 1.0, &reconstruction_error);
  for(index_t i = 0; i < reconstruction_error.n_rows(); i++) {
    
    double row_sum = 0;
    for(index_t j = 0; j < reconstruction_error.n_cols(); j++) {
      row_sum += reconstruction_error.get(i, j) * weights[begin + j];
    }

    expected_maximum_absolute_error_ =
      std::max(expected_maximum_absolute_error_,
	       fabs(row_sum) / ((double) ((*query_leaf_nodes)[i])->count()));
  }

  // The out-going skeleton is constructed from the sampled columns in
  // the matrix factorization.
  outgoing_skeleton_.Init(column_indices.size());
  for(index_t s = 0; s < column_indices.size(); s++) {
    outgoing_skeleton_[s] = tmp_outgoing_skeleton[column_indices[s]];
  }

  // Compute the projection operator, which is the product of the U
  // and the R factor and row scaled by the column scaled C factor.
  Matrix projection_operator;
  
  la::MulInit(u_mat, r_mat, &projection_operator);
  for(index_t i = 0; i < c_mat.n_cols(); i++) {
    
    double scaling_factor = 
      (sample_kernel_matrix.get(0, column_indices[i]) < DBL_EPSILON) ?
      0:c_mat.get(0, i) / sample_kernel_matrix.get(0, column_indices[i]);

    for(index_t j = 0; j < projection_operator.n_cols(); j++) {
      projection_operator.set
	(i, j, projection_operator.get(i, j) * scaling_factor);
    }
  }

  // Compute the outgoing representation by taking the product between
  // the projection operator and the charge distribution vector.
  outgoing_representation_.Init(outgoing_skeleton_.size());
  outgoing_representation_.SetZero();

  for(index_t i = 0; i < end - begin; i++) {

    // Get the weight corresponding to the (i + begin)-th reference
    // point.
    double scaling_factor = weights[i + begin];
    la::AddExpert(projection_operator.n_rows(), scaling_factor,
		  projection_operator.GetColumnPtr(i),
		  outgoing_representation_.ptr());
  }
}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::CombineBasisFunctions
(const MatrixFactorizedFarFieldExpansion &farfield_expansion1,
 const MatrixFactorizedFarFieldExpansion &farfield_expansion2) {

  // The far-field expansion using matrix factorization is formed by
  // concatenating representations.
  const Vector &outgoing_representation1 =
    farfield_expansion1.outgoing_representation();
  const Vector &outgoing_representation2 =
    farfield_expansion2.outgoing_representation();
  const ArrayList<index_t> &outgoing_skeleton1 =
    farfield_expansion1.outgoing_skeleton();
  const ArrayList<index_t> &outgoing_skeleton2 =
    farfield_expansion2.outgoing_skeleton();

  outgoing_representation_.Init(outgoing_representation1.length() + 
				outgoing_representation2.length());
  outgoing_skeleton_.Init(outgoing_skeleton1.size() + 
			  outgoing_skeleton2.size());
  for(index_t i = 0; i < outgoing_representation1.length(); i++) {
    outgoing_representation_[i] = outgoing_representation1[i];
    outgoing_skeleton_[i] = outgoing_skeleton1[i];
  }
  for(index_t i = outgoing_representation1.length(); i < 
	outgoing_representation_.length(); i++) {
    outgoing_representation_[i] = 
      outgoing_representation2[i - outgoing_representation1.length()];
    outgoing_skeleton_[i] = outgoing_skeleton2[i - outgoing_skeleton1.size()];
  }

  // The expected maximum absolute error is basically the sum of the
  // errors for the two expansions.
  expected_maximum_absolute_error_ = 
    farfield_expansion1.expected_maximum_absolute_error() +
    farfield_expansion2.expected_maximum_absolute_error();
}

template<typename TKernelAux>
double MatrixFactorizedFarFieldExpansion<TKernelAux>::EvaluateField
(const Matrix& data, int row_num, int order) const {
  return -1;
}

template<typename TKernelAux>
double MatrixFactorizedFarFieldExpansion<TKernelAux>::EvaluateField
(const Vector &x_q, int order) const {
  return -1;
}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::Init
(const Vector& center, const TKernelAux &ka) {
  
  // Copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  ka_ = &ka;
  
  // Initialize the expected maximum absolute error to zero.
  expected_maximum_absolute_error_ = 0;
}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::Init
(const TKernelAux &ka) {
  
  // Copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);  
  ka_ = &ka;

  // Initialize the expected maximum absolute error to zero.
  expected_maximum_absolute_error_ = 0;
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
  
  // Implement me sometime!
  DEBUG_ASSERT_MSG(false, "Please implement me!");
}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::TranslateToLocal
(MatrixFactorizedLocalExpansion<TKernelAux> &se, int truncation_order,
 const Matrix *reference_set, const Matrix *query_set) const {
  
  // Translating the matrix-factorized far-field moments into local
  // moments involve doing exhaustive evaluation between the points
  // that constitute the incoming skeleton and the outgoing skeleton
  // and multiplying the outgoing representation of the current
  // object.
  const ArrayList<index_t> &incoming_skeleton = se.incoming_skeleton();
  Vector &local_moments = se.coeffs();

  for(index_t q = 0; q < incoming_skeleton.size(); q++) {
    index_t query_point_id = incoming_skeleton[q];
    const double *query_point = query_set->GetColumnPtr(query_point_id);

    for(index_t r = 0; r < outgoing_skeleton_.size(); r++) {
      index_t reference_point_id = outgoing_skeleton_[r];
      const double *reference_point = 
	reference_set->GetColumnPtr(reference_point_id);
      double squared_distance = 
	la::DistanceSqEuclidean(reference_set->n_rows(), query_point, 
				reference_point);
      double kernel_value = 
	(ka_->kernel_).EvalUnnormOnSq(squared_distance);

      // Add the (q, r)-th kernel value times the r-th component of
      // outgoing representation's component to the q-th component of
      // the local moments.
      local_moments[q] += kernel_value * outgoing_representation_[r];
    }
  }
}

#endif
