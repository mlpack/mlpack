/**
 * @file whitening.hpp
 * @author Jeffin Sam
 *
 * Whitening scaling to scale features, Using PCA and ZCA Whitening.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_WHITENING_SCALE_HPP
#define MLPACK_CORE_DATA_WHITENING_SCALE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/lin_alg.hpp>

namespace mlpack {
namespace data {

/**
 * A simple Whitening class.
 *
 * Whitens a matrix using the eigendecomposition of the covariance matrix.
 * Whitening means the covariance matrix of the result is the identity matrix.
 * 
 * For whitening related formula and more info, check the link below.
 * http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
 *
 * @code
 * arma::mat input;
 * Load("train.csv", input);
 * arma::mat output;
 *
 * // Scale the features using PCA.
 * Whitening scale;
 * scale.PCA(input, output);
 *
 * // Scale the features using ZCA.
 * scale.ZCA(input, ouput);
 * @endcode
 */
class Whitening
{
 public:
  /**
  * A constructor to set the regulatization parameter.
  *
  * @param eps Regularization parameter.
  */
  Whitening(double eps = 0.00005)
  {
    epsilon = eps;
  }

  /**
  * Function for PCA whitening.
  *
  * @param input Dataset to scale features.
  * @param output Output matrix with whitened features.
  */
  template<typename MatType>
  void PCA(const MatType& input, MatType& output)
  {
    output.copy_size(input);
    itemMean = arma::mean(input, 1);
    // Centering the Dataset
    output = (input.each_col() - itemMean);
    arma::mat diag;
    // Get eigenvectors and eigenvalues of covariance of input matrix.
    eig_sym(eigenValues, eigenVectors, mlpack::math::ColumnCovariance(output));
    // Generate diagonal matrix using 1 / sqrt(eigenvalues + epsilon) for each 
    // value.
    for (size_t i = 0; i < eigenValues.n_elem; i++)
        eigenValues(i) = eigenValues(i) + epsilon;
    mlpack::math::VectorPower(eigenValues, -0.5);
    diag.zeros(eigenValues.n_elem, eigenValues.n_elem);
    diag.diag() = eigenValues;

    // Our whitening matrix is diag(1 / sqrt(eigenvectors)) * eigenvalues.
    whiteningMatrix = diag * trans(eigenVectors);

    // Now apply the whitening matrix.
    output = whiteningMatrix * output;
  }

  /**
  * Function for ZCA whitening.
  *
  * @param input Dataset to scale features.
  * @param output Output matrix with whitened features.
  */
  template<typename MatType>
  void ZCA(const MatType& input, MatType& output)
  {
    output.copy_size(input);
    itemMean = arma::mean(input, 1);
    // Centering the Dataset
    output = (input.each_col() - itemMean);
    arma::mat diag;
    // Get eigenvectors and eigenvalues of covariance of input matrix.
    eig_sym(eigenValues, eigenVectors, mlpack::math::ColumnCovariance(output));
    // Generate diagonal matrix using 1 / sqrt(eigenvalues + epsilon) for each 
    // value.
    for (size_t i = 0; i < eigenValues.n_elem; i++)
        eigenValues(i) = eigenValues(i) + epsilon;
    mlpack::math::VectorPower(eigenValues, -0.5);
    diag.zeros(eigenValues.n_elem, eigenValues.n_elem);
    diag.diag() = eigenValues;

    // Our whitening matrix is diag(1 / sqrt(eigenvectors)) * eigenvalues.
    whiteningMatrix = diag * trans(eigenVectors);

    // Now apply the whitening matrix.
    output = whiteningMatrix * output;
    // ZCA output.
    output = eigenVectors * output;
  }

  //! Get the Mean row vector.
  const arma::vec& ItemMean() const { return itemMean; }
  //! Get the eigenvalues vector.
  const arma::vec& EigenValues() const { return eigenValues; }
  //! Get the eigenvector.
  const arma::mat& EigenVectors() const { return eigenVectors; }
  //! Get the WhiteningMatrix.
  const arma::mat& WhiteningMatrix() const { return whiteningMatrix; }
  //! Get the Regularisation Parameter.
  const double& Epsilon() const { return epsilon; }

 private:
  // Vector which holds mean of each feature.
  arma::vec itemMean;
  // Mat which hold the eigenvectors.
  arma::mat eigenVectors;
  // Mat which hold the WhiteningMatrix.
  arma::mat whiteningMatrix;
  // Regularization Paramter.
  double epsilon;
  // Vector which hold the eigenvalues.
  arma::vec eigenValues;
}; // class Whitening

} // namespace data
} // namespace mlpack

#endif
