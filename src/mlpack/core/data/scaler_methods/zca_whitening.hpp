/**
 * @file core/data/scaler_methods/zca_whitening.hpp
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
#include <mlpack/core/data/scaler_methods/pca_whitening.hpp>

namespace mlpack {
namespace data {

/**
 * A simple ZCAWhitening class.
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
 * ZCAWhitening scale;
 * scale.Fit(input)
 *
 * // Scale the features.
 * scale.Transform(input, output);
 *
 * // Retransform the input.
 * scale.InverseTransform(output, input);
 * @endcode
 */
class ZCAWhitening
{
 public:
  /**
   * A constructor to set the regularization parameter.
   *
   * @param eps Regularization parameter.
   */
  ZCAWhitening(double eps = 0.00005) : pca(eps) { }

  /**
   * Function to fit features, to find out the min max and scale.
   *
   * @param input Dataset to fit.
   */
  template<typename MatType>
  void Fit(const MatType& input)
  {
    pca.Fit(input);
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
    pca.Transform(input, output);
    output = pca.EigenVectors() * output;
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
    output = inv(pca.EigenVectors()) * arma::diagmat(sqrt(
        pca.EigenValues())) * inv(pca.EigenVectors().t()) * input;
    output = (output.each_col() + pca.ItemMean());
  }

  //! Get the mean row vector.
  const arma::vec& ItemMean() const { return pca.ItemMean(); }
  //! Get the eigenvalues vector.
  const arma::vec& EigenValues() const { return pca.EigenValues(); }
  //! Get the eigenvector.
  const arma::mat& EigenVectors() const { return pca.EigenVectors(); }
  //! Get the regularization parameter.
  double Epsilon() const { return pca.Epsilon(); }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(pca));
  }

 private:
  // A pointer to PcaWhitening Class.
  PCAWhitening pca;
}; // class ZCAWhitening

} // namespace data
} // namespace mlpack

#endif
