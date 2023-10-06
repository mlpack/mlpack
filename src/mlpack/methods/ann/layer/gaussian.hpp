/**
 * @file gaussian_function.hpp
 * @author Himanshu Pathak
 *
 * Definition and implementation of the gaussian function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GAUSSIAN_HPP
#define MLPACK_METHODS_ANN_LAYER_GAUSSIAN_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The gaussian function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& e^{-1 * x^2} \\
 * f'(x) &=& 2 * -x * f(x) 
 * @f}
 */
template <typename MatType = arma::mat>
class GaussianType : public Layer<MatType>
{
 public:
  /**
   * Create the Gaussian object using the specified parameters. 
   */
  GaussianType() {};
  
  virtual ~GaussianType() { }

  //! Copy the other Gaussian layer
  GaussianType(const GaussianType& layer);

  //! Take ownership of the members of the other Gaussian Layer
  GaussianType(GaussianType&& layer);

  //! Copy the other Gaussian layer
  GaussianType& operator=(const GaussianType& layer);

  //! Take ownership of the members of the other Gaussian Layer
  GaussianType& operator=(GaussianType&& layer);

  //! Clone the GaussianType object. This handles polymorphism correctly.
  GaussianType* Clone() const { return new GaussianType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input, const MatType& gy, MatType& g);

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally stored first derivative of the activation function.
  MatType derivative;

}; // class GaussianType

// Convenience typedefs.

// Standard Gaussian layer.
typedef GaussianType<arma::mat> Gaussian;

} // namespace mlpack

// Include implementation.
#include "gaussian_impl.hpp"

#endif