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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the NoisyLinear layer class. It represents a single
 * layer of a neural network, with parametric noise added to its weights.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class NoisyLinear
{
 public:
  //! Create the NoisyLinear object.
  NoisyLinear();

  /**
   * Create the NoisyLinear layer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  NoisyLinear(const size_t inSize,
              const size_t outSize);

  //! Copy constructor.
  NoisyLinear(const NoisyLinear&);

  /*
   * Reset the layer parameter.
   */
  void Reset();

  /*
   * Reset the noise parameters(epsilons).
   */
  void ResetNoise();

  /*
   * Reset the values of layer parameters (factorized gaussian noise).
   */
  void ResetParameters();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the input size.
  size_t InputSize() const { return inSize; }

  //! Get the output size.
  size_t OutputSize() const { return outSize; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! Modify the bias weights of the layer.
  arma::mat& Bias() { return bias; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored weight parameters.
  OutputDataType weight;

  //! Locally-stored weight-mean parameters.
  OutputDataType weightMu;

  //! Locally-stored weight-standard-deviation parameters.
  OutputDataType weightSigma;

  //! Locally-stored weight-epsilon parameters.
  OutputDataType weightEpsilon;

  //! Locally-stored bias parameters.
  OutputDataType bias;

  //! Locally-stored bias-mean parameters.
  OutputDataType biasMu;

  //! Locally-stored bias-standard-deviation parameters.
  OutputDataType biasSigma;

  //! Locally-stored bias-epsilon parameters.
  OutputDataType biasEpsilon;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class NoisyLinear

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "noisylinear_impl.hpp"

#endif
