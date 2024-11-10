/**
 * @file methods/ann/layer/multiply_constant.hpp
 * @author Marcus Edel
 *
 * Definition of the MultiplyConstantLayer class, which multiplies the input by
 * a (non-learnable) constant.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the multiply constant layer. The multiply constant layer
 * multiplies the input by a (non-learnable) constant.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class MultiplyConstantType : public Layer<InputType, OutputType>
{
 public:
  //! Create the MultiplyConstant object.
  MultiplyConstantType(const double scalar = 1.0);

  //! Clone the MultiplyConstantType object. This handles polymorphism
  //! correctly.
  MultiplyConstantType* Clone() const
  {
    return new MultiplyConstantType(*this);
  }

  //! Copy Constructor.
  MultiplyConstant(const MultiplyConstant& layer);

  //! Move Constructor.
  MultiplyConstant(MultiplyConstant&& layer);

  //! Copy assignment operator.
  MultiplyConstant& operator=(const MultiplyConstant& layer);

  //! Move assignment operator.
  MultiplyConstant& operator=(MultiplyConstant&& layer);

  /**
   * Ordinary feed forward pass of a neural network. Multiply the input with the
   * specified constant scalar value.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network. The backward pass
   * multiplies the error with the specified constant scalar value.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  //! Get the scalar multiplier.
  double Scalar() const { return scalar; }
  //! Modify the scalar multiplier.
  double& Scalar() { return scalar; }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored constant scalar value.
  double scalar;
}; // class MultiplyConstantType

// Convenience typedefs.

// Standard MultiplyConstant layer.
using MultiplyConstant = MultiplyConstantType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "multiply_constant_impl.hpp"

#endif
