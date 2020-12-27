/**
 * @file methods/ann/layer/linear_no_bias.hpp
 * @author Marcus Edel
 *
 * Definition of the LinearNoBias class also known as fully-connected layer or
 * affine transformation without the bias term.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

#include "layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the LinearNoBias class. The LinearNoBias class represents a
 * single layer of a neural network.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *    cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *    to also be in this type. The type also allows the computation and weight
 *    type to differ from the input type (Default: arma::mat).
 * @tparam RegularizerType Type of the regularizer to be used (Default no
 *    regularizer).
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class LinearNoBiasType : public Layer<InputType, OutputType>
{
 public:
  //! Create the LinearNoBias object.
  LinearNoBiasType();

  /**
   * Create the LinearNoBias object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param regularizer The regularizer to use, optional.
   */
  LinearNoBiasType(const size_t inSize,
                   const size_t outSize,
                   RegularizerType regularizer = RegularizerType());

  //! Reset the layer parameter.
  void Reset();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }


  //! Get the input size.
  size_t InputSize() const { return inSize; }

  //! Get the output size.
  size_t OutputSize() const { return outSize; }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored weight parameter.
  OutputType weight;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class LinearNoBiasType

// Convenience typedefs.

// Standard Linear without bias layer using no regularization.
typedef LinearNoBiasType<arma::mat, arma::mat, NoRegularizer> LinearNoBias;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "linear_no_bias_impl.hpp"

#endif
