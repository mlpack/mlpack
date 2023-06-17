/**
 * @file methods/ann/layer/star_relu.hpp
 * @author Mayank Raj
 *
 * Definition of StarReLU layer introduced by Yu et al.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_STAR_RELU_HPP
#define MLPACK_METHODS_ANN_LAYER_STAR_RELU_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The StarReLU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& s \cdot (\text{ReLU}(x))^2 + b
 * f'(x) &=& s \cdot 2 \cdot \text{ReLU}(x)
 * @f}
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class StarReLUType : public Layer<MatType>
{
 public:
  /**
   * Create the StarReLU object using the specified parameters.
   * The constants s and b can be adjusted via the corresponding parameters.
   * The default values of s and b are s=0.8944 and b=-0.4472.
   * @param s Scaling constant.
   * @param b Bias constant.
   */
  StarReLUType(const double s = 0.8944, const double b = -0.4472);

  //! Clone the StarReLUType object. This handles polymorphism correctly.
  StarReLUType* Clone() const { return new StarReLUType(*this); }

  // Virtual destructor.
  virtual ~StarReLUType() { }

  //! Copy the given StarReLUType.
  StarReLUType(const StarReLUType& other);
  //! Take ownership of the given StarReLUType.
  StarReLUType(StarReLUType&& other);
  //! Copy the given StarReLUType.
  StarReLUType& operator=(const StarReLUType& other);
  //! Take ownership of the given StarReLUType.
  StarReLUType& operator=(StarReLUType&& other);

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

  //! Get the scaling constant.
  double const& Scale() const { return s; }
  //! Modify the scaling constant.
  double& Scale() { return s; }

  //! Get the bias constant.
  double const& Bias() const { return b; }
  //! Modify the bias constant.
  double& Bias() { return b; }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Scaling constant.
  double s;
  //! Bias constant.
  double b;
}; // class StarReLUType

// Convenience typedefs.
typedef StarReLUType<arma::mat> StarReLU;

} // namespace mlpack

// Include implementation.
#include "star_relu_impl.hpp"

#endif