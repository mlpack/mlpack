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

/**
 * Implementation of the LinearNoBias class. The LinearNoBias class represents a
 * single layer of a neural network.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 * @tparam RegularizerType Type of the regularizer to be used (Default no
 *    regularizer).
 */
template<
    typename MatType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class LinearNoBiasType : public Layer<MatType>
{
 public:
  //! Create the LinearNoBias object.
  LinearNoBiasType();

  /**
   * Create the LinearNoBias object using the specified number of units.
   *
   * @param outSize The number of output units.
   * @param regularizer The regularizer to use, optional.
   */
  LinearNoBiasType(const size_t outSize,
                   RegularizerType regularizer = RegularizerType());

  //! Clone the LinearNoBiasType object. This handles polymorphism correctly.
  LinearNoBiasType* Clone() const { return new LinearNoBiasType(*this); }

  //! Reset the layer parameter.
  void SetWeights(const MatType& weightsIn);

  //! Copy constructor.
  LinearNoBiasType(const LinearNoBiasType& layer);

  //! Move constructor.
  LinearNoBiasType(LinearNoBiasType&&);

  //! Copy assignment operator.
  LinearNoBiasType& operator=(const LinearNoBiasType& layer);

  //! Move assignment operator.
  LinearNoBiasType& operator=(LinearNoBiasType&& layer);

  //! Virtual destructor.
  virtual ~LinearNoBiasType() { }

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
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient);

  //! Get the parameters.
  const MatType& Parameters() const { return weight; }
  //! Modify the parameters.
  MatType& Parameters() { return weight; }

  //! Get the number of weights in the layer.
  size_t WeightSize() const { return inSize * outSize; }

  //! Compute the output dimensions of the layer using `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight parameter.
  MatType weight;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class LinearNoBiasType

// Convenience typedefs.

// Standard Linear without bias layer using no regularization.
using LinearNoBias = LinearNoBiasType<arma::mat, NoRegularizer>;

} // namespace mlpack

// Include implementation.
#include "linear_no_bias_impl.hpp"

#endif
