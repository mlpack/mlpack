#ifndef SUBSPACE_STAT_H
#define SUBSPACE_STAT_H

#include "fastlib/fastlib.h"

class SubspaceStat {
  
 private:

  static const double epsilon_ = 0.01;

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

  void FastSvdByColumnSampling_(const Matrix& mean_centered, bool transposed) {

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
    int num_samples = max(mean_centered.n_cols() / 2, 1);
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
		  &right_singular_vectors_);
      singular_values_.Init(eigen_count);
      for(index_t i = 0; i < eigen_count; i++) {
	singular_values_[i] = sqrt(tmp_eigen_values[i]);
	la::Scale(mean_centered.n_rows(), 1.0 / singular_values_[i],
		right_singular_vectors_.GetColumnPtr(i));
      }
      
      // Now compute the right singular vectors from the left singular
      // vectors.
      la::MulTransAInit(mean_centered, right_singular_vectors_,
			&left_singular_vectors_);
      for(index_t i = 0; i < left_singular_vectors_.n_cols(); i++) {
	double length = 
	  la::LengthEuclidean(left_singular_vectors_.n_rows(),
			      left_singular_vectors_.GetColumnPtr(i));
	la::Scale(left_singular_vectors_.n_rows(), 1.0 / length,
		  left_singular_vectors_.GetColumnPtr(i));
      }
    }
    else {
      // Now exploit the relationship between the right and the left
      // singular vectors. Normalize and retrieve the singular values.
      la::MulInit(sampled_columns, aliased_right_singular_vectors,
		  &left_singular_vectors_);
      singular_values_.Init(eigen_count);
      for(index_t i = 0; i < eigen_count; i++) {
	singular_values_[i] = sqrt(tmp_eigen_values[i]);
	la::Scale(mean_centered.n_rows(), 1.0 / singular_values_[i],
		left_singular_vectors_.GetColumnPtr(i));
      }
      
      // Now compute the right singular vectors from the left singular
      // vectors.
      la::MulTransAInit(mean_centered, left_singular_vectors_,
			&right_singular_vectors_);
      for(index_t i = 0; i < right_singular_vectors_.n_cols(); i++) {
	double length = 
	  la::LengthEuclidean(right_singular_vectors_.n_rows(),
			      right_singular_vectors_.GetColumnPtr(i));
	la::Scale(right_singular_vectors_.n_rows(), 1.0 / length,
		  right_singular_vectors_.GetColumnPtr(i));
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
    
    // Determine which dimension is longer: the row or the column...
    // If there are more columns than rows, then we do
    // column-sampling.
    if(dataset.n_rows() <= count) {
      
      Matrix mean_centered;

      // Compute the mean vector owned by this node.
      ComputeColumnMean_(dataset, start, count, &mean_vector_);
      
      // Compute the mean centered dataset.
      ColumnMeanCenter_(dataset, start, count, mean_vector_, &mean_centered);
   
      FastSvdByColumnSampling_(mean_centered, false);
    }
    else {

      Matrix mean_centered;

      // Compute the mean vector owned by this node.
      ComputeColumnMean_(dataset, start, count, &mean_vector_);
      
      // Compute the mean centered dataset.
      ColumnMeanCenterTranspose_(dataset, start, count, mean_vector_, 
				 &mean_centered);

      FastSvdByColumnSampling_(mean_centered, true);
    }
    exit(0);
  }

  /** Merge two eigenspaces into one.
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const SubspaceStat& left_stat, const SubspaceStat& right_stat) {

  }

  SubspaceStat() { }

  ~SubspaceStat() { }

};

#endif
