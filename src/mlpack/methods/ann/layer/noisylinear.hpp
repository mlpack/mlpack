/**
 * @file methods/ann/layer/noisylinear.hpp
 * @author Nishant Kumar
 *
 * Definition of the NoisyLinear layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NOISYLINEAR_HPP
#define MLPACK_METHODS_ANN_LAYER_NOISYLINEAR_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the NoisyLinear layer class. It represents a single
 * layer of a neural network, with parametric noise added to its weights.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class NoisyLinearType : public Layer<MatType>
{
 public:
  /**
   * Create the NoisyLinear layer object using the specified number of units.
   *
   * @param outSize The number of output units.
   */
  NoisyLinearType(const size_t outSize = 0);

  //! Clone the NoisyLinearType object. This handles polymorphism correctly.
  NoisyLinearType* Clone() const { return new NoisyLinearType(*this); }

  // Virtual destructor.
  virtual ~NoisyLinearType() { }

  //! Copy the given NoisyLinear layer (but not weights).
  NoisyLinearType(const NoisyLinearType& other);
  //! Take ownership of the given NoisyLinear layer (but not weights).
  NoisyLinearType(NoisyLinearType&& other);
  //! Copy the given NoisyLinear layer (but not weights).
  NoisyLinearType& operator=(const NoisyLinearType& other);
  //! Take ownership of the given NoisyLinear layer (but not weights).
  NoisyLinearType& operator=(NoisyLinearType&& other);

  //! Reset the layer parameter.
  void SetWeights(const MatType& weightsIn);

  //! Reset the noise parameters (epsilons).
  void ResetNoise();

  //! Reset the values of layer parameters (factorized gaussian noise).
  void ResetParameters();

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
  MatType const& Parameters() const { return weights; }
  //! Modify the parameters.
  MatType& Parameters() { return weights; }

  //! Get the shape of the input.
  //! Modify the bias weights of the layer.
  MatType& Bias() { return bias; }

  //! Compute the number of parameters in the layer.
  size_t WeightSize() const { return (outSize * inSize + outSize) * 2; }

  //! Compute the output dimensions of the layer given `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally stored number of input units.
  size_t inSize;

  //! Locally-stored weight object.
  MatType weights;

  //! Locally-stored weight parameters.
  MatType weight;

  //! Locally-stored weight-mean parameters.
  MatType weightMu;

  //! Locally-stored weight-standard-deviation parameters.
  MatType weightSigma;

  //! Locally-stored weight-epsilon parameters.
  MatType weightEpsilon;

  //! Locally-stored bias parameters.
  MatType bias;

  //! Locally-stored bias-mean parameters.
  MatType biasMu;

  //! Locally-stored bias-standard-deviation parameters.
  MatType biasSigma;

  //! Locally-stored bias-epsilon parameters.
  MatType biasEpsilon;
}; // class NoisyLinearType

// Convenience typedefs.

// Standard noisy linear layer.
using NoisyLinear = NoisyLinearType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "noisylinear_impl.hpp"

#endif
