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
  int num_reference_samples = (int) sqrt(end - begin);
  int num_query_samples = (int) query_leaf_nodes->size();
  
  // Allocate a temporary space for holding the indices of the
  // reference points, from which the outgoing skeleton will be
  // chosen.
  ArrayList<index_t> tmp_outgoing_skeleton;
  tmp_outgoing_skeleton.Init(num_reference_samples);
  for(index_t r = 0; r < num_reference_samples; r++) {

    // Choose a random reference point and record its index.
    index_t random_reference_point_index = math::RandInt(begin, end);
    tmp_outgoing_skeleton[r] = random_reference_point_index;
  }
  // Sort the chosen reference indices and eliminate duplicates...
  qsort(tmp_outgoing_skeleton.begin(), tmp_outgoing_skeleton.size(),
	sizeof(index_t), &qsort_compar_);
  remove_duplicates_in_sorted_array_(tmp_outgoing_skeleton);
  num_reference_samples = tmp_outgoing_skeleton.size();

  // After determining the number of reference samples to take,
  // allocate the space for the sample kernel matrix to be computed.
  sample_kernel_matrix.Init(num_query_samples, num_reference_samples);

  for(index_t r = 0; r < num_reference_samples; r++) {

    // The reference point...
    const double *reference_point =
      reference_set.GetColumnPtr(tmp_outgoing_skeleton[r]);

    for(index_t c = 0; c < num_query_samples; c++) {
      
      // Choose a random query point from the current query strata...
      index_t random_query_point_index =
	math::RandInt(((*query_leaf_nodes)[c])->begin(),
		      ((*query_leaf_nodes)[c])->end());
      const double *query_point =
	query_set->GetColumnPtr(random_query_point_index);	

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
      projection_operator.set(i, j, projection_operator.get(i, j) *
			      scaling_factor);
    }
  }

  // Compute the outgoing representation by taking the product between
  // the projection operator and the charge distribution vector.
  outgoing_representation_.Init(outgoing_skeleton_.size());
  outgoing_representation_.SetZero();
  for(index_t i = 0; i < outgoing_skeleton_.size(); i++) {
    double scaling_factor = 
      weights[outgoing_skeleton_[i]] *
      (((double) end - begin) / ((double) outgoing_skeleton_.size()));
    la::AddExpert(projection_operator.n_rows(), scaling_factor,
		  projection_operator.GetColumnPtr(i),
		  outgoing_representation_.ptr());
  }

  // Turn on the flag that says the object has been initialized.
  is_initialized_ = true;
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
  center_.Copy(center);
  sea_ = &(ka.sea_);
  ka_ = &ka;

  // The default flag for initialization is false.
  is_initialized_ = false;
}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::Init
(const TKernelAux &ka) {
  
  // Copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);  
  sea_ = &(ka.sea_);
  center_.Init(sea_->get_dimension());
  center_.SetZero();
  ka_ = &ka;
  
  // The default flag for initialization is false.
  is_initialized_ = false;
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
  
  // The far-field expansion using matrix factorization is formed by
  // concatenating representations.
  if(is_initialized_) {
    Vector tmp_outgoing_representation;
    const Vector &outgoing_representation_to_be_translated =
      se.outgoing_representation();
    const ArrayList<index_t> &outgoing_skeleton_to_be_translated =
      se.outgoing_skeleton();
    tmp_outgoing_representation.Copy(outgoing_representation_);
    outgoing_representation_.Destruct();    
    outgoing_representation_.Init
      (tmp_outgoing_representation.length() +
       outgoing_representation_to_be_translated.length());
    for(index_t i = 0; i < tmp_outgoing_representation.length(); i++) {
      outgoing_representation_[i] = tmp_outgoing_representation[i];
    }
    for(index_t i = tmp_outgoing_representation.length(); i < 
	  outgoing_representation_.length(); i++) {
      outgoing_representation_[i] = 
	outgoing_representation_to_be_translated
	[i - tmp_outgoing_representation.length()];
    }
    for(index_t i = 0; i < outgoing_skeleton_to_be_translated.size(); i++) {
      outgoing_skeleton_.PushBackCopy(outgoing_skeleton_to_be_translated[i]);
    }
    
    for(index_t i = 0; i < outgoing_representation_.length(); i++) {
      printf("%g ", outgoing_representation_[i]);
    }
    printf("\n");
    for(index_t i = 0; i < outgoing_skeleton_.size(); i++) {
      printf("%d ", outgoing_skeleton_[i]);
    }
    printf("\n");
  }
  else {
    outgoing_representation_.Copy(se.outgoing_representation());
    outgoing_skeleton_.InitCopy(se.outgoing_skeleton());
  }
  
  // Turn on the initialization flag on.
  is_initialized_ = true;
}

template<typename TKernelAux>
void MatrixFactorizedFarFieldExpansion<TKernelAux>::TranslateToLocal
(MatrixFactorizedLocalExpansion<TKernelAux> &se, int truncation_order,
 const Matrix *reference_set, const Matrix *query_set) {
  
  // Translating the matrix-factorized far-field moments into local
  // moments involve doing exhaustive evaluation between the points
  // that constitute the incoming skeleton and the outgoing skeleton
  // and multiplying the outgoing representation of the current
  // object.
  const ArrayList<index_t> &incoming_skeleton = se.incoming_skeleton();
  Vector &local_moments = se.incoming_representation();

  for(index_t q = 0; q < incoming_skeleton.size(); q++) {
    index_t query_point_id = incoming_skeleton[q];
    const double *query_point = query_set->GetColumnPtr(query_point_id);

    for(index_t r = 0; r < outgoing_skeleton_.size(); r++) {
      index_t reference_point_id = outgoing_skeleton_[r];
      const double *reference_point = 
	reference_set->GetColumnPtr(reference_point_id);
      double squared_distance = 
	la::DistanceSqEuclidean(reference_set->n_rows(),
				query_point, reference_point);
      double kernel_value = 
	(ka_->kernel_).EvalUnnormOnSq(squared_distance);

      // Add the (q, r)-th kernel value times the r-th component of
      // outgoing representation's component to the r-th component of
      // the local moments.
      local_moments[r] += kernel_value * outgoing_representation_[r];
    }
  }
}

#endif
