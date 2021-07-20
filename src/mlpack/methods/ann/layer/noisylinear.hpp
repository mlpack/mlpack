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
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the NoisyLinear layer class. It represents a single
 * layer of a neural network, with parametric noise added to its weights.
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *    cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *    to also be in this type. The type also allows the computation and weight
 *    type to differ from the input type (Default: arma::mat).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class NoisyLinearType : public Layer<InputType, OutputType>
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

  //! Reset the layer parameter.
  void SetWeights(typename OutputType::elem_type* weightsPtr);

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

  //! Modify the bias weights of the layer.
  OutputType& Bias() { return bias; }

  size_t WeightSize() const { return (outSize * inSize + outSize) * 2; }

  void ComputeOutputDimensions()
  {
    inSize = std::accumulate(this->inputDimensions.begin(),
        this->inputDimensions.end(), 0);
    this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
        1);

    // The NoisyLinear layer flattens its output.
    this->outputDimensions[0] = outSize;
  }

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Cached size of input.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored weight parameters.
  OutputType weight;

  //! Locally-stored weight-mean parameters.
  OutputType weightMu;

  //! Locally-stored weight-standard-deviation parameters.
  OutputType weightSigma;

  //! Locally-stored weight-epsilon parameters.
  OutputType weightEpsilon;

  //! Locally-stored bias parameters.
  OutputType bias;

  //! Locally-stored bias-mean parameters.
  OutputType biasMu;

  //! Locally-stored bias-standard-deviation parameters.
  OutputType biasSigma;

  //! Locally-stored bias-epsilon parameters.
  OutputType biasEpsilon;

}; // class NoisyLinearType

// Convenience typedefs.

// Standard noisy linear layer.
typedef NoisyLinearType<arma::mat, arma::mat> NoisyLinear;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "noisylinear_impl.hpp"

#endif
