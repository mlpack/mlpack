/**
 * @file core/data/scaler_methods/pca_whitening.hpp
 * @author Jeffin Sam
 *
 * Whitening scaling to scale features, Using PCA Whitening.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_PCA_WHITENING_SCALE_HPP
#define MLPACK_CORE_DATA_PCA_WHITENING_SCALE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/ccov.hpp>

namespace mlpack {
namespace data {

/**
 * A simple PCAWhitening class.
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
 * // Fit the features.
 * PCAWhitening scale;
 * scale.Fit(input)
 *
 * // Scale the features.
 * scale.Transform(input, output);
 *
 * // Retransform the input.
 * scale.InverseTransform(output, input);
 * @endcode
 */
class PCAWhitening
{
 public:
  /**
   * A constructor to set the regularization parameter.
   *
   * @param eps Regularization parameter.
   */
  PCAWhitening(double eps = 0.00005)
  {
    epsilon = eps;
    // Ensure scaleMin is smaller than scaleMax.
    if (epsilon < 0)
    {
      throw std::runtime_error("Regularization parameter is not correct");
    }
  }

  /**
   * Function to fit features, to find out the min max and scale.
   *
   * @param input Dataset to fit.
   */
  template<typename MatType>
  void Fit(const MatType& input)
  {
    itemMean = arma::mean(input, 1);
    // Get eigenvectors and eigenvalues of covariance of input matrix.
    eig_sym(eigenValues, eigenVectors, ColumnCovariance(
        input.each_col() - itemMean));
    eigenValues += epsilon;
  }

  /**
   * Function for PCA whitening.
   *
   * @param input Dataset to scale features.
   * @param output Output matrix with whitened features.
   */
  template<typename MatType>
  void Transform(const MatType& input, MatType& output)
  {
    if (eigenValues.is_empty() || eigenVectors.is_empty())
    {
      throw std::runtime_error("Call Fit() before Transform(), please"
          " refer to the documentation.");
    }
    output.copy_size(input);
    output = (input.each_col() - itemMean);
    output = arma::diagmat(1.0 / (sqrt(eigenValues))) * eigenVectors.t()
        * output;
  }

  /**
   * Function to retrieve original dataset.
   *
   * @param input Scaled dataset.
   * @param output Output matrix with original Dataset.
   */
  template<typename MatType>
  void InverseTransform(const MatType& input, MatType& output)
  {
    output = arma::diagmat(sqrt(eigenValues)) * inv(eigenVectors.t())
        * input;
    output = (output.each_col() + itemMean);
  }

  //! Get the mean row vector.
  const arma::vec& ItemMean() const { return itemMean; }
  //! Get the eigenvalues vector.
  const arma::vec& EigenValues() const { return eigenValues; }
  //! Get the eigenvector.
  const arma::mat& EigenVectors() const { return eigenVectors; }
  //! Get the regularization parameter.
  const double& Epsilon() const { return epsilon; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(eigenValues));
    ar(CEREAL_NVP(eigenVectors));
    ar(CEREAL_NVP(itemMean));
    ar(CEREAL_NVP(epsilon));
  }

 private:
  // Vector which holds mean of each feature.
  arma::vec itemMean;
  // Mat which hold the eigenvectors.
  arma::mat eigenVectors;
  // Regularization Paramter.
  double epsilon;
  // Vector which hold the eigenvalues.
  arma::vec eigenValues;
}; // class PCAWhitening

} // namespace data
} // namespace mlpack

#endif
