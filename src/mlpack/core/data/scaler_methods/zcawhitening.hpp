/**
 * @file zcawhitening.hpp
 * @author Jeffin Sam
 *
 * Whitening scaling to scale features, Using ZCA Whitening.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_ZCA_WHITENING_SCALE_HPP
#define MLPACK_CORE_DATA_ZCA_WHITENING_SCALE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/lin_alg.hpp>
#include <mlpack/core/data/scaler_methods/pcawhitening.hpp>

namespace mlpack {
namespace data {

/**
 * A simple ZcaWhitening class.
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
 * // Scale the features using ZCA.
 * ZcaWhitening scale;
 * scale.Transform(input, output);
 * @endcode
 */
class ZcaWhitening
{
 public:
  /**
  * A constructor to set the regulatization parameter.
  *
  * @param eps Regularization parameter.
  */
  ZcaWhitening(double eps = 0.00005)
  {
    epsilon = eps;
  }

  /**
  * Function for ZCA whitening.
  *
  * @param input Dataset to scale features.
  * @param output Output matrix with whitened features.
  */
  template<typename MatType>
  void Transform(const MatType& input, MatType& output)
  {
    data::PcaWhitening scale(epsilon);
    scale.Transform(input, output);
    eigenVectors = scale.EigenVectors();
    itemMean = scale.ItemMean();
    whiteningMatrix = scale.WhiteningMatrix();
    eigenValues = scale.EigenValues(); 
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
}; // class ZcaWhitening

} // namespace data
} // namespace mlpack

#endif
