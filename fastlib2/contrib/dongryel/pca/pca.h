#ifndef PCA_H
#define PCA_H

#include "fastlib/fastlib.h"

class Pca {

 public:

  static void GramSchmidt(const Matrix *current_bases, 
			  int num_current_components, Vector &new_basis,
			  double epsilon) {
    
    // Make a backup copy of the new basis.
    Vector new_basis_copy;    
    new_basis_copy.Copy(new_basis);
    
    for(index_t i = 0; i < num_current_components; i++) {
     
      // Get a pointer to the i-th previous basis.
      Vector previous_basis;
      current_bases->MakeColumnVector(i, &previous_basis);

      // Compute the dot-product between the new basis and the
      // previous basis.
      double dot_product = la::Dot(new_basis_copy, previous_basis);
      
      // Subtract off the component described the i-th previous basis.
      la::AddExpert(-dot_product, previous_basis, &new_basis);      
    }

    // Normalize the new basis to be of unit norm.
    la::Scale(1.0 / la::LengthEuclidean(new_basis), &new_basis);

  }

  static void ComputeCovariance(const Matrix &data, Matrix *covariance) {

    // Compute the mean vector.
    Vector mean;
    ComputeMean(data, &mean);

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

  static void ComputeMean(const Matrix &data, index_t start, index_t count,
			  Vector *mean) {
    mean->Init(data.n_rows());
    mean->SetZero();

    for(index_t i = start; i < start + count; i++) {
      Vector data_col;
      data.MakeColumnVector(i, &data_col);
      la::AddTo(data_col, mean);
    }
    la::Scale(1.0 / ((double) count), mean);    
  }

  static void RandomUnitVector(Vector &random_unit_vector) {
    
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

  static void MeanCenter(const Matrix &data, Matrix *data_copy) {
    
    // Compute the column mean.
    Vector mean;
    ComputeMean(data, &mean);

    data_copy->Init(data.n_rows(), data.n_cols());
    
    // Subtract the mean vector from each column of the matrix.
    for(index_t i = 0; i < data.n_cols(); i++) {
      Vector data_copy_col, data_col;
      data_copy->MakeColumnVector(i, &data_copy_col);
      data.MakeColumnVector(i, &data_col);

      la::SubOverwrite(data_col, mean, &data_copy_col);
    }
  }

  static void MeanCenter(const Matrix &data, const Vector &mean,
			 Matrix *data_copy) {
    
    data_copy->Init(data.n_rows(), data.n_cols());
    
    // Subtract the mean vector from each column of the matrix.
    for(index_t i = 0; i < data.n_cols(); i++) {
      Vector data_copy_col, data_col;
      data_copy->MakeColumnVector(i, &data_copy_col);
      data.MakeColumnVector(i, &data_col);

      la::SubOverwrite(data_col, mean, &data_copy_col);
    }
  }

  static void MeanCenter(const Matrix &data, index_t start, index_t count,
			 const Vector &mean, Matrix *data_copy) {
    
    data_copy->Init(data.n_rows(), count);
    
    // Subtract the mean vector from each column of the matrix.
    for(index_t i = start; i < start + count; i++) {
      Vector data_copy_col, data_col;
      data_copy->MakeColumnVector(i - start, &data_copy_col);
      data.MakeColumnVector(i, &data_col);

      la::SubOverwrite(data_col, mean, &data_copy_col);
    }
  }

  static void PrincipalAngles(const Matrix &left_basis,
			      const Matrix &right_basis,
			      Matrix *left_basis_principal_vectors,
			      Matrix *right_basis_principal_vectors,
			      Vector *principal_angles) {
    
    // Compute the QR decompositions of the two basis sets.
    Matrix left_basis_q_factor, left_basis_r_factor;
    Matrix right_basis_q_factor, right_basis_r_factor;
    la::QRInit(left_basis, &left_basis_q_factor, &left_basis_r_factor);
    la::QRInit(right_basis, &right_basis_q_factor, &right_basis_r_factor);
    
    // Compute the product of the two q-factors and SVD of the product.
    Matrix prod_q_factors, left_singular_vectors, right_singular_vectors;
    la::MulTransAInit(left_basis_q_factor, right_basis_q_factor,
		      &prod_q_factors);
    la::SVDInit(prod_q_factors, principal_angles, &left_singular_vectors,
		&right_singular_vectors);
    la::MulInit(left_basis_q_factor, left_singular_vectors, 
		left_basis_principal_vectors);
    la::MulTransBInit(right_basis_q_factor, right_singular_vectors, 
		      right_basis_principal_vectors);
  }

  /** @brief Computes the principal components of the given dataset by
   *         computing the right singular vector using SVD.
   */
  static void SvdLeftSingularVector(const Matrix &data,
				    Vector *eigen_values,
				    Matrix *principal_components) {

    // Mean-center the dataset and compute its left singular vector
    // (eigenvector).
    Matrix mean_centered;
    MeanCenter(data, &mean_centered);
    
    // Singular values and singular vectors.
    Matrix right_singular_vectors;
    
    la::SVDInit(mean_centered, eigen_values, principal_components,
		&right_singular_vectors);
    for(index_t i = 0; i < eigen_values->length(); i++) {
      (*eigen_values)[i] = ((*eigen_values)[i] * (*eigen_values)[i]) / 
	((double) data.n_cols());
    }    
  }

  /** @brief Computes the principal components of the given dataset
   *         using the usual eigendecomposition of the covariance
   *         matrix.
   */
  static void EigenDecomposeCovariance(const Matrix &data,
				       Vector *eigen_values,
				       Matrix *principal_components) {

    // First, compute the covariance matrix of the dataset.
    Matrix covariance;
    ComputeCovariance(data, &covariance);
    
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
				  int num_components_desired,
				  double epsilon) {
    
    // First, compute the covariance matrix of the dataset.
    Matrix covariance;
    ComputeCovariance(data, &covariance);

    // Allocate enough space for storing the principal components and
    // eigenvalues.
    principal_components->Init(data.n_rows(), num_components_desired);
    principal_components->SetZero();
    eigen_values->Init(num_components_desired);
    eigen_values->SetZero();
    
    // Temporary space for storing the product.
    Vector product;
    product.Init(data.n_rows());

    for(index_t c = 0; c < num_components_desired; c++) {
      
      Vector previous_iteration_current_basis;
      Vector current_basis;
      principal_components->MakeColumnVector(c, &current_basis);      
      
      // Generate random unit vector.
      RandomUnitVector(current_basis);

      // Set the previous iteration's vector basis to be zero vector.
      previous_iteration_current_basis.Init(current_basis.length());
      previous_iteration_current_basis.SetZero();

      // Flag for convergence detection.
      bool converged = false;

      // Repeat until convergence.
      do {

	// Compute the product of the current basis and the covariance
	// matrix.
	la::MulOverwrite(covariance, current_basis, &product);
	(*eigen_values)[c] = la::LengthEuclidean(product);

	previous_iteration_current_basis.CopyValues(current_basis);
	current_basis.CopyValues(product);	

	// Orthogonalize the product against all existing basis.
	GramSchmidt(principal_components, c, current_basis, 
		    epsilon);

	if(fabs(la::Dot(current_basis, 
			previous_iteration_current_basis) - 1) < epsilon ) {
	  converged = true;
	}
      } while(!converged);
      
    } // end of iterating over components
  }

};

#endif
