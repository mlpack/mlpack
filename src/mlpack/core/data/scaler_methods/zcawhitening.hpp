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
  * A constructor to set the regulatization parameter.
  *
  * @param eps Regularization parameter.
  */
  ZcaWhitening(double eps = 0.00005)
  {
    epsilon = eps;
  }
  /**
  * Function to fit features, to find out the min max and scale.
  *
  * @param input Dataset to fit.
  */
  template<typename MatType>
  void Fit(const MatType& input)
  {
    data::PcaWhitening scale(epsilon);
    scale.Fit(input);
    // Important to store results, so that i can use in InverseTranform().
    eigenVectors = scale.EigenVectors();
    itemMean = scale.ItemMean();
    eigenValues = scale.EigenValues();
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
    output.copy_size(input);
    output = (input.each_col() - itemMean);
    output = arma::diagmat(1.0 / (arma::sqrt(eigenValues))) * eigenVectors.t()
        * output;
    output = eigenVectors * output;
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
    output = inv(eigenVectors) * arma::diagmat(arma::sqrt(eigenValues))
        * inv(eigenVectors.t()) * input;
    output = (output.each_col() + itemMean);
  }

  //! Get the Mean row vector.
  const arma::vec& ItemMean() const { return itemMean; }
  //! Get the eigenvalues vector.
  const arma::vec& EigenValues() const { return eigenValues; }
  //! Get the eigenvector.
  const arma::mat& EigenVectors() const { return eigenVectors; }
  //! Get the Regularisation Parameter.
  const double& Epsilon() const { return epsilon; }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(eigenValues);
    ar & BOOST_SERIALIZATION_NVP(eigenVectors);
    ar & BOOST_SERIALIZATION_NVP(itemMean);
    ar & BOOST_SERIALIZATION_NVP(epsilon);
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
}; // class ZcaWhitening

} // namespace data
} // namespace mlpack

#endif
