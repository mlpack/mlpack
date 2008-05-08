#ifndef SUBSPACE_STAT_H
#define SUBSPACE_STAT_H

#include "fastlib/fastlib.h"

class SubspaceStat {
  
 private:

  static const double epsilon_ = 0.01;

  static void ComputeResidualBasis_(const Matrix &first_basis,
				    const Matrix &second_basis,
				    Matrix *residual_basis) {
    
    printf("First basis: %d %d\n", first_basis.n_rows(), first_basis.n_cols());
    printf("Second basis: %d %d\n", second_basis.n_rows(),
	   second_basis.n_cols());

    // First project second basis onto the first basis...
    Matrix tmp_matrix;
    la::MulTransAInit(first_basis, second_basis, &tmp_matrix);
    
    // Compute the residual...
    residual_basis->Copy(second_basis);

    la::MulExpert(-1, false, first_basis, false, tmp_matrix, 1, 
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

      if(length > DBL_EPSILON) {
	la::Scale(residual_basis->n_rows(), 1.0 / length, column_vector);
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
    Init(dataset, start, count);
  }

  SubspaceStat() { }

  ~SubspaceStat() { }

};

#endif
