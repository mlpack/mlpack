#ifndef LANCZOS_PCA_H
#define LANCZOS_PCA_H

#include "fastlib/fastlib.h"

class LanczosPca {

 private:

  static void ComputeCovariance(const Matrix &data, const Vector &mean, 
				Matrix *covariance) {

    covariance->Init(data.n_rows(), data.n_rows());
    covariance->SetZero();

    for(index_t i = 0; i < data.n_cols(); i++) {
      for(index_t c = 0; c < data.n_rows(); c++) {
	for(index_t r = 0; r < data.n_rows(); r++) {
	  
	  covariance->set(r, c, covariance->get(r, c) + data.get(c, i) * data.get(r, i));
	}
      }      
    }
    la::Scale(1.0 / ((double) data.n_cols()), covariance);
  }

 public:

  /** @brief Computes the principal components of the given dataset.
   */
  static void Compute(const Matrix &data, Matrix *principal_components) {

    // Compute the mean of the column vectors.
    Vector mean_vector;
    mean_vector.Init(data.n_rows());
    mean_vector.SetZero();

    for(index_t i = 0; i < data.n_cols(); i++) {
      Vector data_col;
      data.MakeColumnVector(i, &data_col);
      la::AddTo(data_col, &mean_vector);
    }
    la::Scale(1.0 / ((double) data.n_cols()), &mean_vector);

    // First, compute the covariance matrix of the dataset.
    Matrix covariance;
    ComputeCovariance(data, mean_vector, &covariance);
  }

};

#endif
