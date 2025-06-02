/**
 * @file methods/ann/layer/linear.hpp
 * @author Marcus Edel
 *
 * Definition of the Linear layer class also known as fully-connected layer or
 * affine transformation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Linear layer class. The Linear class represents a
 * single layer of a neural network.
 *
 * The linear layer applies a linear transformation to the incoming data
 * (input), i.e. y = Ax + b. The input matrix given in Forward(input, output)
 * must be either a vector or matrix. If the input is a matrix, then each column
 * is assumed to be an input sample of given batch.
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
class LinearType : public Layer<MatType>
{
 public:
  //! Create the Linear object.
  LinearType();

  /**
   * Create the Linear layer object with the specified number of output
   * dimensions.
   *
   * @param outSize The output dimension.
   * @param regularizer The regularizer to use, optional (default: no
   *     regularizer).
   */
  LinearType(const size_t outSize,
             RegularizerType regularizer = RegularizerType());

  virtual ~LinearType() { }

  //! Clone the LinearType object. This handles polymorphism correctly.
  LinearType* Clone() const { return new LinearType(*this); }

  //! Copy the other Linear layer (but not weights).
  LinearType(const LinearType& layer);

  //! Take ownership of the members of the other Linear layer (but not weights).
  LinearType(LinearType&& layer);

  //! Copy the other Linear layer (but not weights).
  LinearType& operator=(const LinearType& layer);

  //! Take ownership of the members of the other Linear layer (but not weights).
  LinearType& operator=(LinearType&& layer);

  /**
   * Reset the layer parameter (weights and bias). The method is called to
   * assign the allocated memory to the internal learnable parameters.
   */
  void SetWeights(const MatType& weightsIn);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * f(x) is a linear transformation: Ax + b, where x is the given input, x are
   * the layer weights and b is the layer bias.
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
   * To compute the downstream gradient (g) the chain rule is used.
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
  const MatType& Parameters() const { return weights; }
  //! Modify the parameters.
  MatType& Parameters() { return weights; }

  //! Get the weight of the layer.
  MatType const& Weight() const { return weight; }
  //! Modify the weight of the layer.
  MatType& Weight() { return weight; }

  //! Get the bias of the layer.
  MatType const& Bias() const { return bias; }
  //! Modify the bias weights of the layer.
  MatType& Bias() { return bias; }

  //! Get the size of the weights.
  size_t WeightSize() const { return (inSize * outSize) + outSize; }

  //! Compute the output dimensions of the layer given `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.  This holds all the weights in a vectorized
  //! form; i.e., the weight and the bias.
  MatType weights;

  //! Locally-stored weight parameters.
  MatType weight;

  //! Locally-stored bias term parameters.
  MatType bias;

  //! Locally-stored regularizer object.
  RegularizerType regularizer;
}; // class LinearType

// Convenience typedefs.

// Standard Linear layer using no regularization.
using Linear = LinearType<arma::mat, NoRegularizer>;

} // namespace mlpack

// Include implementation.
#include "linear_impl.hpp"

#endif
