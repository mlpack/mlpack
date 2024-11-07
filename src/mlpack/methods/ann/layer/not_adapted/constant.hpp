/**
 * @file methods/ann/layer/constant.hpp
 * @author Marcus Edel
 *
 * Definition of the Constant class, which outputs a constant value given
 * any input.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONSTANT_HPP
#define MLPACK_METHODS_ANN_LAYER_CONSTANT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the constant layer. The constant layer outputs a given
 * constant value given any input value.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class ConstantType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create an empty Constant layer.
   */
  ConstantType();

  /**
   * Create the Constant object that outputs a given constant scalar value
   * given any input value.
   *
   * @param outSize The number of output units.
   * @param scalar The constant value used to create the constant output.
   */
  ConstantType(const size_t outSize, const double scalar = 0);

  //! Copy another ConstantType.
  ConstantType(const ConstantType& layer);
  //! Take ownership of another ConstantType.
  ConstantType(ConstantType&& layer);
  //! Copy another ConstantType.
  ConstantType& operator=(const ConstantType& layer);
  //! Take ownership of another ConstantType.
  ConstantType& operator=(ConstantType&& layer);

  //! Clone the ConstantType object. This handles polymorphism correctly.
  ConstantType* Clone() const { return new ConstantType(*this); }

  /**
   * Ordinary feed forward pass of a neural network. The forward pass fills the
   * output with the specified constant parameter.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network. The backward pass of the
   * constant layer is returns always a zero output error matrix.
   *
   * @param * (input) The propagated input activation.
   * @param * (gy) The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& /* gy */,
                OutputType& g);

  //! Get the output size.
  const std::vector<size_t>& OutputDimensions() const
  {
    std::vector<size_t> result(this->inputDimensions.size(), 0);
    result[0] = outSize;
    return result;
  }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored constant output matrix.
  OutputType constantOutput;
}; // class ConstantType

// Convenience typedefs.

// Standard HardShrink layer.
using Constant = ConstantType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "constant_impl.hpp"

#endif
