#ifndef PCA_H
#define PCA_H

#include "fastlib/fastlib.h"

class Pca {

 private:

  static void ComputeCovariance_(const Matrix &data, Matrix *covariance) {

    // Compute the mean vector.
    Vector mean;
    ComputeMean_(data, &mean);

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

  static void ComputeMean_(const Matrix &data, Vector *mean) {
    mean->Init(data.n_rows());
    mean->SetZero();

    for(index_t i = 0; i < data.n_cols(); i++) {
      Vector data_col;
      data.MakeColumnVector(i, &data_col);
      la::AddTo(data_col, mean);
    }
    la::Scale(1.0 / ((double) data.n_cols()), mean);    
  }

  static void RandomUnitVector_(Vector &random_unit_vector) {
    
    bool done = false;
    double length;

    while(!done) {

      for(index_t i = 0; i < random_unit_vector.length(); i++) {
	random_unit_vector[i] = -1 + 2 * math::Random();
      }
      if((length = la::LengthEuclidean(random_unit_vector)) < 1 &&
	 length > 0) {
	done = true;
      }
    }
    
    la::Scale(1.0 / length, &random_unit_vector);
  }

 public:

  /** @brief Computes the principal components of the given dataset
   *         using the usual eigendecomposition of the covariance
   *         matrix.
   */
  static void EigenDecomposeCovariance(const Matrix &data,
				       Vector *eigen_values,
				       Matrix *principal_components) {

    // First, compute the covariance matrix of the dataset.
    Matrix covariance;
    ComputeCovariance_(data, &covariance);
    
    // Compute the eigenvalues/eigenvectors of the covariance matrix.
    la::EigenvectorsInit(covariance, eigen_values, principal_components);
  }

  /** @brief Computes the principal components of the given dataset
   *         using the fixed-point algorithm.
   *
   *  Alok Sharma and Kuldip K. Paliwal. Fast Principal Component
   *  Analysis Using Fixed-point Algorithm. Pattern Recognition
   *  Letters 28 (2007) 1151--1155.
   */
  static void FixedPointAlgorithm(const Matrix &data,
				  Vector *eigen_values,
				  Matrix *principal_components,
				  double epsilon) {
    
    // First, compute the covariance matrix of the dataset.
    Matrix covariance;
    ComputeCovariance_(data, &covariance);

    // The current number of principal components to be searched.
    int current_num_components = 1;

    do {
      

    } while(tolerance > epsilon);
  }

};

#endif
