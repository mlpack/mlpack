#ifndef PCA_H
#define PCA_H

#include "fastlib/fastlib.h"

class Pca {

 private:

  static void ComputeCovariance(const Matrix &data, const Vector &mean, 
				Matrix *covariance) {

    // Allocate D by D covariance matrix and initialize it to zero.
    covariance->Init(data.n_rows(), data.n_rows());
    covariance->SetZero();
    
    for(index_t i = 0; i < data.n_cols(); i++) {
      for(index_t c = 0; c < data.n_rows(); c++) {
	for(index_t r = 0; r < data.n_rows(); r++) {	  
	  covariance->set(r, c, covariance->get(r, c) + 
			  (data.get(c, i) - mean[c]) * 
			  (data.get(r, i) - mean[r]));
	} // end of iterating over each row of the covariance matrix.
      } // end of iterating over each column of the covariance matrix.
    } // end of iterating over each data point
    la::Scale(1.0 / ((double) data.n_cols()), covariance);
  }

  static void ComputeMean(const Matrix &data, Vector *mean) {
    mean->Init(data.n_rows());
    mean->SetZero();

    for(index_t i = 0; i < data.n_cols(); i++) {
      Vector data_col;
      data.MakeColumnVector(i, &data_col);
      la::AddTo(data_col, mean);
    }
    la::Scale(1.0 / ((double) data.n_cols()), mean);    
  }

 public:

  /** @brief Computes the principal components of the given dataset.
   */
  static void EigenDecomposeCovariance(const Matrix &data,
				       Vector *eigen_values,
				       Matrix *principal_components) {

    // Compute the mean of the column vectors.
    Vector mean_vector;
    ComputeMean(data, &mean_vector);

    // First, compute the covariance matrix of the dataset.
    Matrix covariance;
    ComputeCovariance(data, mean_vector, &covariance);
    
    // Compute the eigenvalues/eigenvectors of the covariance matrix.
    la::EigenvectorsInit(covariance, eigen_values, principal_components);
  }

};

#endif
