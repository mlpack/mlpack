/**
 * @file methods/ann/layer/linear_recurrent.hpp
 * @author Ryan Curtin
 *
 * Definition of the LinearRecurrent class, which implements the most basic
 * possible linear recurrent layer.  (This is the first thing you learn in the
 * intro to RNNs lecture.)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_RECURRENT_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_RECURRENT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The LinearRecurrent layer defines a dense layer of simple recurrent neurons.
 * If a layer has n neurons, then a layer stores a hidden state of size n (one
 * hidden element for each layer), and the output of the layer is defined as
 *
 * ```
 * f(X) = WX + UH + b
 * ```
 *
 * for trainable parameters `W` and `b` (a matrix of weights and vector of
 * biases, just like a regular dense linear layer), and `U`, which is the
 * learned weights for the previous time step's hidden state.  `H`, the hidden
 * state, is just `f(X)` at the previous time step.
 *
 * Note that no nonlinearity is applied before the output is fed back into the
 * layer.
 */
template<
    typename MatType = arma::mat,
    typename RegularizerType = NoRegularizer
>
class LinearRecurrentType : public RecurrentLayer<MatType>
{
 public:
  /**
   * Create the LinearRecurrent layer.
   */
  LinearRecurrentType();

  /**
   * Create the LinearRecurrent layer object with the specified number of
   * output dimensions (e.g. neurons).
   *
   * @param outSize The output dimension.
   * @param regularizer The regularizer to use; optional (default: no
   *    regularizer)
   */
  LinearRecurrentType(const size_t outSize,
                      RegularizerType regularizer = RegularizerType());

  virtual ~LinearRecurrentType() { }

  // Clone the LinearRecurrentType layer.  This handles polymorphism correctly.
  LinearRecurrentType* Clone() const { return new LinearRecurrentType(*this); }

  // Copy the other linear recurrent layer, including hidden recurrent state
  // (but not weights).
  LinearRecurrentType(const LinearRecurrentType& layer);

  // Take ownership of the members of the other linear recurrent layer,
  // including hidden recurrent state (but not weights).
  LinearRecurrentType(LinearRecurrentType&& layer);

  // Copy the other linear recurrent layer, including hidden recurrent state
  // (but not weights).
  LinearRecurrentType& operator=(const LinearRecurrentType& layer);

  // Take ownership of the members of the other linear recurrent layer,
  // including hidden recurrent state (but not weights).
  LinearRecurrentType& operator=(LinearRecurrentType&& layer);

  /**
   * Set the parameters of the layer (weights, hidden state weights, and bias).
   * This method is called by the network to assign allocated memory to the
   * internal learnable parameters.
   */
  void SetWeights(const MatType& weightsIn);

  /**
   * Forward pass of linear recurrent layer.
   * Computes f(X) = WX + UH + b, where H is the current recurrent hidden state.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Backward pass of linear recurrent layer.
   */
  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g);

  /**
   * Compute the gradient of the weights in the layer with respect to the given
   * input and error.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient);

  //! Get the parameters.
  const MatType& Parameters() const { return parameters; }
  //! Modify the parameters.
  MatType& Parameters() { return parameters; }

  // Get the (non-recurrent) weight matrix (W).
  const MatType& Weights() const { return weights; }
  // Modify the (non-recurrent) weight matrix (W).
  MatType& Weights() { return weights; }

  // Get the recurrent state weight matrix (U).
  const MatType& RecurrentWeights() const { return recurrentWeights; }
  // Modify the recurrent state weight matrix (U).
  MatType& RecurrentWeights() { return recurrentWeights; }

  // Get the bias vector (b).
  const MatType& Bias() const { return bias; }
  // Modify the bias vector (b).
  MatType& Bias() { return bias; }

  // Get the total number of trainable parameters.
  size_t WeightSize() const;

  // Get the total number of recurrent elements.
  size_t RecurrentSize() const;

  // Compute the output dimensions of the layer, assuming that inputDimensions
  // has been set.
  void ComputeOutputDimensions();

  // Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  // Locally-stored number of input neurons.
  size_t inSize;
  // Locally-stored number of output neurons.
  size_t outSize;

  // This holds all of the trainable weights of the layer.
  MatType parameters;

  // Weight matrix for inputs.
  MatType weights;
  // Weight matrix for hidden state.
  MatType recurrentWeights;
  // Bias vector.
  MatType bias;

  // Locally-stored regularizer object.
  RegularizerType regularizer;
};

// Convenience typedefs.

using LinearRecurrent = LinearRecurrentType<arma::mat, NoRegularizer>;

} // namespace mlpack

#include "linear_recurrent_impl.hpp"

#endif
