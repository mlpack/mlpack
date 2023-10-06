/**
 * @file methods/ann/layer/gelu_function.hpp
 * @author Vivek Pal
 * @author Dakshit Agrawal
 *
 * Definition and implementation of the Gaussian Error Linear Unit (GELU)
 * function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GELU_HPP
#define MLPACK_METHODS_ANN_LAYER_GELU_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The GELU function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = 0.5 * x * {1 + tanh[(2/pi)^(1/2) * (x + 0.044715 * x^3)]} \\
 * f'(x) = 0.5 * tanh(0.0356774 * x^3) + 0.797885 * x) + 
 *         (0.0535161x^3 + 0.398942 * x) * 
 *         sech^2(0.0356774 * x^3+0.797885 * x) + 0.5\\
 * @f}
 */
template <typename MatType = arma::mat>
class GELUType : public Layer<MatType>
{
 public:
  /**
   * Create the GELU object using the specified parameters. 
   */
  GELUType() {};
  
  virtual ~GELUType() { }

  //! Copy the other GELU layer
  GELUType(const GELUType& layer);

  //! Take ownership of the members of the other GELU Layer
  GELUType(GELUType&& layer);

  //! Copy the other GELU layer
  GELUType& operator=(const GELUType& layer);

  //! Take ownership of the members of the other GELU Layer
  GELUType& operator=(GELUType&& layer);

  //! Clone the GELUType object. This handles polymorphism correctly.
  GELUType* Clone() const { return new GELUType(*this); }

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

}; // class GELUType

// Convenience typedefs.

// Standard GELU layer.
typedef GELUType<arma::mat> GELU;

} // namespace mlpack

// Include implementation.
#include "gelu_impl.hpp"

#endif