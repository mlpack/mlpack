/**
 * @file zca_whitening.hpp
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
#include <mlpack/core/data/scaler_methods/pca_whitening.hpp>

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
 * // Fit the features.
 * ZcaWhitening scale;
 * scale.Fit(input)
 *
 * // Scale the features.
 * scale.Transform(input, output);
 *
 * // Retransform the input.
 * scale.InverseTransform(output, input);
 * @endcode
 */
class ZcaWhitening
{
 public:
  /**
  * A constructor to set the regularization parameter.
  *
  * @param eps Regularization parameter.
  */
  ZcaWhitening(double eps = 0.00005)
  {
    zca = new data::PcaWhitening(eps);
  }
  /**
  * Function to fit features, to find out the min max and scale.
  *
  * @param input Dataset to fit.
  */
  template<typename MatType>
  void Fit(const MatType& input)
  {
    zca->Fit(input);
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
    zca->Transform(input, output);
    output = zca->EigenVectors() * output;
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
    output = inv(zca->EigenVectors()) * arma::diagmat(arma::sqrt(
        zca->EigenValues())) * inv(zca->EigenVectors().t()) * input;
    output = (output.each_col() + zca->ItemMean());
  }

  //! Get the Mean row vector.
  const arma::vec& ItemMean() const { return zca->ItemMean(); }
  //! Get the eigenvalues vector.
  const arma::vec& EigenValues() const { return zca->EigenValues(); }
  //! Get the eigenvector.
  const arma::mat& EigenVectors() const { return zca->EigenVectors(); }
  //! Get the Regularisation Parameter.
  const double& Epsilon() const { return zca->Epsilon(); }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(zca);
  }

 private:
  // A pointer to PcaWhitening Class.
  PcaWhitening* zca;
}; // class ZcaWhitening

} // namespace data
} // namespace mlpack

#endif
