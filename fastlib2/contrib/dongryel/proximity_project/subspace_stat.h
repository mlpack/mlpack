#ifndef SUBSPACE_STAT_H
#define SUBSPACE_STAT_H

#include "fastlib/fastlib.h"

class SubspaceStat {
  
 private:

  static const double epsilon_ = 0.01;

  static void ComputeResidualBasis_(const Matrix &first_basis,
				    const Matrix &second_basis,
				    const Vector &mean_diff,
				    Matrix *projection,
				    Matrix *residual_basis) {

    // First project second basis and the difference of the two means
    // onto the first basis...
    Matrix projection_first_basis;
    Vector projection_mean_diff;
    projection->Init(first_basis.n_cols(), second_basis.n_cols() + 1);
    projection_first_basis.Alias(projection->GetColumnPtr(0),
				 first_basis.n_cols(), second_basis.n_cols());
    projection_mean_diff.Alias(projection->GetColumnPtr(second_basis.n_cols()),
			       projection->n_rows());
    la::MulTransAOverwrite(first_basis, second_basis, &projection_first_basis);
    la::MulOverwrite(mean_diff, first_basis, &projection_mean_diff);
    
    // Reconstruct and compute the reconstruction error...    
    residual_basis->Init(second_basis.n_rows(), second_basis.n_cols() + 1);
    Matrix residual_basis_second_basis;
    Vector residual_basis_mean_diff;
    residual_basis_second_basis.Alias(residual_basis->GetColumnPtr(0),
				      first_basis.n_rows(),
				      second_basis.n_cols());
    residual_basis_mean_diff.Alias(residual_basis->GetColumnPtr
				   (second_basis.n_cols()),
				   first_basis.n_rows());
    residual_basis_second_basis.CopyValues(second_basis);
    residual_basis_mean_diff.CopyValues(mean_diff);

    la::MulExpert(-1, false, first_basis, false, *projection, 1, 
		  residual_basis);

    // Loop over each residual basis...
    for(index_t i = 0; i < residual_basis->n_cols(); i++) {
      
      double *column_vector = residual_basis->GetColumnPtr(i);

      for(index_t j = 0; j < i; j++) {
	double dot_product = la::Dot(residual_basis->n_rows(), column_vector,
				     residual_basis->GetColumnPtr(j));
	la::AddExpert(residual_basis->n_rows(), -dot_product,
		      residual_basis->GetColumnPtr(j), column_vector);
      }
      
      // Normalize the vector if done...
      double length = la::LengthEuclidean(residual_basis->n_rows(),
					  column_vector);

      if(length > epsilon_) {
	la::Scale(residual_basis->n_rows(), 1.0 / length, column_vector);
      }
      else {
	la::Scale(residual_basis->n_rows(), 0, column_vector);
      }
    } // end of looping over each residual basis...
  }

  static void ComputeColumnMean_(const Matrix &data, index_t start, 
				 index_t count, Vector *mean) {

    mean->Init(data.n_rows());
    mean->SetZero();
    
    for(index_t i = start; i < start + count; i++) {
      Vector data_col;
      data.MakeColumnVector(i, &data_col);
      la::AddTo(data_col, mean);
    }
    la::Scale(1.0 / ((double) count), mean);    
  }
  
  static void ColumnMeanCenter_(const Matrix &data, index_t start, 
				index_t count, const Vector &mean, 
				Matrix *data_copy) {
    
    data_copy->Init(data.n_rows(), count);
    
    // Subtract the mean vector from each column of the matrix.
    for(index_t i = start; i < start + count; i++) {
      Vector data_copy_col, data_col;
      data_copy->MakeColumnVector(i - start, &data_copy_col);
      data.MakeColumnVector(i, &data_col);

      la::SubOverwrite(data_col, mean, &data_copy_col);
    }
  }
  
  static void ColumnMeanCenterTranspose_(const Matrix &data, index_t start, 
					 index_t count, const Vector &mean, 
					 Matrix *data_copy) {
    
    data_copy->Init(count, data.n_rows());
    
    // Subtract the mean vector from each column of the matrix.
    for(index_t i = start; i < start + count; i++) {
      for(index_t j = 0; j < data.n_rows(); j++) {
	data_copy->set(i - start, j, data.get(j, i) - mean[j]);
      }
    }
  }

  static void ComputeNormalizedCumulativeDistribution_(const Vector &src,
						       Vector *dest,
						       double *total) {
    *total = 0;
    dest->Init(src.length());
    (*dest)[0] = src[0];
    (*total) += src[0];
    
    for(index_t i = 1; i < src.length(); i++) {
      (*dest)[i] = (*dest)[i - 1] + src[i];
      (*total) += src[i];
    }
  }

  static index_t FindBinNumber_(const Vector &cumulative_distribution, 
				double random_number) {

    for(index_t i = 0; i < cumulative_distribution.length(); i++) {
      if(random_number < cumulative_distribution[i]) {
	return i;
      }
    }
    return cumulative_distribution.length() - 1;
  }

  void FastSvdByColumnSampling_(const Matrix& mean_centered, bool transposed,
				Vector &singular_values_arg,
				Matrix &left_singular_vectors_arg,
				Matrix &right_singular_vectors_arg) {

    // First determine the column length-squared distribution...
    Vector squared_lengths;
    squared_lengths.Init(mean_centered.n_cols());
    for(index_t i = 0; i < mean_centered.n_cols(); i++) {
      squared_lengths[i] = 
	la::Dot(mean_centered.n_rows(), mean_centered.GetColumnPtr(i),
		mean_centered.GetColumnPtr(i));
    }

    // Compute the normalized cumulative distribution on the squared lengths.
    Vector normalized_cumulative_squared_lengths;
    double total_squared_lengths;
    ComputeNormalizedCumulativeDistribution_
      (squared_lengths, &normalized_cumulative_squared_lengths,
       &total_squared_lengths);
    
    // The number of samples...
    int num_samples = std::max((int) sqrt(mean_centered.n_cols()), 1);
    Matrix sampled_columns;
    sampled_columns.Init(mean_centered.n_rows(), num_samples);

    // Commence sampling...
    for(index_t s = 0; s < num_samples; s++) {
      
      // Generate random number between 0 and total_squared_lengths
      // and find out which column is picked.
      double random_number = math::Random(0, total_squared_lengths);
      index_t sample_number = 
	FindBinNumber_(normalized_cumulative_squared_lengths, random_number);
      
      // Normalize proportion to squared length and the number of
      // samples.
      double probability =
	squared_lengths[sample_number] / total_squared_lengths;
      for(index_t j = 0; j < mean_centered.n_rows(); j++) {
	sampled_columns.set(j, s, mean_centered.get(j, sample_number) /
			    sqrt(num_samples * probability));
      }
    }

    // Let C = sampled columns. Then here we compute C^T C and
    // computes its eigenvector.
    Matrix sampled_product, tmp_right_singular_vectors, tmp_vectors;
    Vector tmp_eigen_values;
    la::MulTransAInit(sampled_columns, sampled_columns, &sampled_product);
    la::SVDInit(sampled_product, &tmp_eigen_values, 
		&tmp_right_singular_vectors, &tmp_vectors);

    // Cut off small eigen values...
    int eigen_count = 0;
    for(index_t i = 0; i < tmp_eigen_values.length(); i++) {
      if(tmp_eigen_values[i] >= 0 &&
	 tmp_eigen_values[i] >= epsilon_ * tmp_eigen_values[0]) {
	eigen_count++;
      }
    }
    Matrix aliased_right_singular_vectors;
    aliased_right_singular_vectors.Alias
      (tmp_right_singular_vectors.GetColumnPtr(0),
       tmp_right_singular_vectors.n_rows(), eigen_count);

    if(transposed) {
      // Now exploit the relationship between the right and the left
      // singular vectors. Normalize and retrieve the singular values.
      la::MulInit(sampled_columns, aliased_right_singular_vectors,
		  &right_singular_vectors_arg);
      singular_values_arg.Init(eigen_count);
      for(index_t i = 0; i < eigen_count; i++) {
	singular_values_arg[i] = sqrt(tmp_eigen_values[i]);
	la::Scale(mean_centered.n_rows(), 1.0 / singular_values_arg[i],
		  right_singular_vectors_arg.GetColumnPtr(i));
      }
      
      // Now compute the right singular vectors from the left singular
      // vectors.
      la::MulTransAInit(mean_centered, right_singular_vectors_arg,
			&left_singular_vectors_arg);
      for(index_t i = 0; i < left_singular_vectors_arg.n_cols(); i++) {
	double length = 
	  la::LengthEuclidean(left_singular_vectors_arg.n_rows(),
			      left_singular_vectors_arg.GetColumnPtr(i));
	la::Scale(left_singular_vectors_arg.n_rows(), 1.0 / length,
		  left_singular_vectors_arg.GetColumnPtr(i));
      }
    }
    else {
      // Now exploit the relationship between the right and the left
      // singular vectors. Normalize and retrieve the singular values.
      la::MulInit(sampled_columns, aliased_right_singular_vectors,
		  &left_singular_vectors_arg);
      singular_values_arg.Init(eigen_count);
      for(index_t i = 0; i < eigen_count; i++) {
	singular_values_arg[i] = sqrt(tmp_eigen_values[i]);
	la::Scale(mean_centered.n_rows(), 1.0 / singular_values_arg[i],
		left_singular_vectors_arg.GetColumnPtr(i));
      }
      
      // Now compute the right singular vectors from the left singular
      // vectors.
      la::MulTransAInit(mean_centered, left_singular_vectors_arg,
			&right_singular_vectors_arg);
      for(index_t i = 0; i < right_singular_vectors_arg.n_cols(); i++) {
	double length = 
	  la::LengthEuclidean(right_singular_vectors_arg.n_rows(),
			      right_singular_vectors_arg.GetColumnPtr(i));
	la::Scale(right_singular_vectors_arg.n_rows(), 1.0 / length,
		  right_singular_vectors_arg.GetColumnPtr(i));
      }
    }
  }

 public:
  
  int start_;
  
  int count_;
  
  Vector mean_vector_;

  Matrix left_singular_vectors_;
  
  Vector singular_values_;

  Matrix right_singular_vectors_;

  /** @brief Compute PCA exhaustively for leaf nodes.
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count) {

    // Set the start and count info before anything else...
    start_ = start;
    count_ = count;

    if(count == 1) {
      mean_vector_.Init(dataset.n_rows());
      mean_vector_.SetZero();
      left_singular_vectors_.Init(dataset.n_rows(), 1);
      left_singular_vectors_.SetZero();
      singular_values_.Init(1);
      singular_values_.SetZero();
      right_singular_vectors_.Init(count, 1);
      right_singular_vectors_.SetZero();
      return;
    }

    Matrix mean_centered;
    
    // Compute the mean vector owned by this node.
    ComputeColumnMean_(dataset, start, count, &mean_vector_);

    // Determine which dimension is longer: the row or the column...
    // If there are more columns than rows, then we do
    // column-sampling.
    if(dataset.n_rows() <= count) {
            
      // Compute the mean centered dataset.
      ColumnMeanCenter_(dataset, start, count, mean_vector_, &mean_centered);
   
      FastSvdByColumnSampling_(mean_centered, false, singular_values_,
			       left_singular_vectors_, 
			       right_singular_vectors_);
    }
    else {
      
      // Compute the mean centered dataset.
      ColumnMeanCenterTranspose_(dataset, start, count, mean_vector_, 
				 &mean_centered);

      FastSvdByColumnSampling_(mean_centered, true, singular_values_,
			       left_singular_vectors_,
			       right_singular_vectors_);
    }
  }

  /** Merge two eigenspaces into one.
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const SubspaceStat& left_stat, const SubspaceStat& right_stat) {

    // Set the starting index and the count...
    start_ = start;
    count_ = count;
    
    // Compute the weighted mean of the two means...
    mean_vector_.Copy(left_stat.mean_vector_);
    la::Scale(left_stat.count_, &mean_vector_);
    la::AddExpert(right_stat.count_, right_stat.mean_vector_, &mean_vector_);
    la::Scale(1.0 / ((double) count), &mean_vector_);
    
    // Compute the difference between the two PCA models...
    Vector diff_mean;
    la::SubInit(left_stat.mean_vector_, right_stat.mean_vector_, &diff_mean);
    
    // Compute the residual of the projection of the right basis and
    // the mean difference onto the left basis.
    Matrix subspace_projection_reconstruction_error, subspace_projection;
    ComputeResidualBasis_(left_stat.left_singular_vectors_,
			  right_stat.left_singular_vectors_, diff_mean,
			  &subspace_projection,
			  &subspace_projection_reconstruction_error);
    
    // Now we setup the eigenproblem to be solved for stitching two
    // PCA models together.
    Matrix merging_problem;
    int dimension_merging_problem = left_stat.singular_values_.length() +
      right_stat.singular_values_.length();
    merging_problem.Init(dimension_merging_problem, dimension_merging_problem);
    merging_problem.SetZero();
    
    // Compute the multiplicative factors.
    double factor1 = ((double) left_stat.count_) / ((double) count);
    double factor2 = ((double) right_stat.count_) / ((double) count);
    double factor3 = ((double) left_stat.count_ * right_stat.count_) /
      ((double) count * count);
    
    // Setup the top left block...using the outer-product formulation
    // of the matrix-matrix product.
    //
    // Remember that eigenvalues are squared singular values!!
    for(index_t j = 0; j < left_stat.singular_values_.length(); j++) {
      merging_problem.set(j, j, factor1 * left_stat.singular_values_[j] *
			  left_stat.singular_values_[j] / 
			  ((double) left_stat.count_));
    }

    for(index_t j = 0; j < right_stat.singular_values_.length() + 1; j++) {
      const double *column_vector = subspace_projection.GetColumnPtr(j);

      // Now loop over each component of the upper left submatrix.
      for(index_t i = 0; i < left_stat.singular_values_.length(); i++) {
	for(index_t k = 0; k < left_stat.singular_values_.length(); k++) {

	  if(j < right_stat.singular_values_.length()) {
	    merging_problem.set(k, i, merging_problem.get(k, i) + factor2 *
				column_vector[k] * column_vector[i] *
				right_stat.singular_values_[j] *
				right_stat.singular_values_[j] /
				((double) right_stat.count_));
	  }
	  else {
	    merging_problem.set(k, i, merging_problem.get(k, i) + factor3 *
				column_vector[k] * column_vector[i]);
	  }
	}
      }
    }
    
    // Compute the projection of the right basis and the mean
    // difference onto the residual basis.
    Matrix projection_right_basis;
    la::MulTransAInit(subspace_projection_reconstruction_error, 
		      right_stat.left_singular_vectors_, 
		      &projection_right_basis);
    Vector projection_mean_diff;
    la::MulInit(diff_mean, subspace_projection_reconstruction_error,
		&projection_mean_diff);

    // Set up the top right block...also using the outer-product
    // formulation of the matrix-matrix product.
    for(index_t j = 0; j < right_stat.singular_values_.length() + 1; j++) {
      const double *column_vector = subspace_projection.GetColumnPtr(j);
      const double *column_vector2 = 
	(j < right_stat.singular_values_.length()) ?
	projection_right_basis.GetColumnPtr(j):projection_mean_diff.ptr();

      // Now loop over each component of the upper right submatrix.
      for(index_t i = left_stat.singular_values_.length(); 
	  i < dimension_merging_problem; i++) {
	for(index_t k = 0; k < left_stat.singular_values_.length(); k++) {

	  if(j < right_stat.singular_values_.length()) {
	    merging_problem.set
	      (k, i, merging_problem.get(k, i) + factor2 *
	       column_vector[k] * 
	       column_vector2[i - left_stat.singular_values_.length()] *
	       right_stat.singular_values_[j] *
	       right_stat.singular_values_[j] / ((double) right_stat.count_));
	  }
	  else {
	    merging_problem.set
	      (k, i, merging_problem.get(k, i) + factor3 *
	       column_vector[k] * 
	       column_vector2[i - left_stat.singular_values_.length()]);
	  }
	} // end of iterating over each row...
      } // end of iterating over each column...
    } // end of iterating over each column...
    
    // Set up the lower left block... This is basically a tranpose of
    // the upper right block.
    for(index_t i = 0; i < left_stat.singular_values_.length(); i++) {
      for(index_t j = left_stat.singular_values_.length();
	  j < dimension_merging_problem; j++) {
	merging_problem.set(j, i, merging_problem.get(i, j));
      }
    }
    // Set up the lower right block... also using the outer-product
    // formulation of the matrix-matrix product.
    for(index_t j = 0; j < right_stat.singular_values_.length() + 1; j++) {
      const double *column_vector = 
	(j < right_stat.singular_values_.length()) ?
	projection_right_basis.GetColumnPtr(j):projection_mean_diff.ptr();
      
      // Now loop over each component of the upper left submatrix.
      for(index_t i = left_stat.singular_values_.length(); 
	  i < dimension_merging_problem; i++) {
	for(index_t k = left_stat.singular_values_.length(); 
	    k < dimension_merging_problem; k++) {

	  if(j < right_stat.singular_values_.length()) {
	    merging_problem.set
	      (k, i, merging_problem.get(k, i) + factor2 *
	       column_vector[k - left_stat.singular_values_.length()] * 
	       column_vector[i - left_stat.singular_values_.length()] *
	       right_stat.singular_values_[j] * 
	       right_stat.singular_values_[j] / ((double) right_stat.count_));
	  }
	  else {
	    merging_problem.set
	      (k, i, merging_problem.get(k, i) + factor3 *
	       column_vector[k - left_stat.singular_values_.length()] * 
	       column_vector[i - left_stat.singular_values_.length()]);
	  }
	}
      }
    }
    
    // Compute the eigenvector of the system and rotate...
    Vector tmp_singular_values;
    Matrix tmp_left_singular_vectors, tmp_right_singular_vectors;

    la::SVDInit(merging_problem, &tmp_singular_values,
		&tmp_left_singular_vectors, &tmp_right_singular_vectors);
    int eigen_count = 0;
    for(index_t i = 0; i < tmp_singular_values.length(); i++) {
      if(tmp_singular_values[i] >= epsilon_ * tmp_singular_values[0]) {
	eigen_count++;
      }
    }

    // Rotation...
    left_singular_vectors_.Init(dataset.n_rows(), eigen_count);
    left_singular_vectors_.SetZero();

    for(index_t i = 0; i < tmp_left_singular_vectors.n_cols(); i++) {
      const double *column_vector =
	(i < left_stat.left_singular_vectors_.n_cols()) ?
	left_stat.left_singular_vectors_.GetColumnPtr(i):
	subspace_projection_reconstruction_error.GetColumnPtr
	(i - left_stat.left_singular_vectors_.n_cols());

      for(index_t j = 0; j < eigen_count; j++) {
	for(index_t k = 0; k < dataset.n_rows(); k++) {
	  left_singular_vectors_.set
	    (k, j, left_singular_vectors_.get(k, j) +
	     column_vector[k] * tmp_left_singular_vectors.get(i, j));
	}
      }
    }
    
    // Copy over the singular values...
    singular_values_.Init(eigen_count);
    for(index_t i = 0; i < eigen_count; i++) {
      singular_values_[i] = sqrt(tmp_singular_values[i] * count_);
    }
    
    // Initialize to dummy values...
    right_singular_vectors_.Init(0, 0);
  }

  SubspaceStat() { }

  ~SubspaceStat() { }

};

#endif
