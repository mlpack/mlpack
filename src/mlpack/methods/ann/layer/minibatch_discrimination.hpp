/**
 * @file methods/ann/layer/minibatch_discrimination.hpp
 * @author Saksham Bansal
 *
 * Definition of the MiniBatchDiscrimination layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_HPP
#define MLPACK_METHODS_ANN_LAYER_MINIBATCH_DISCRIMINATION_HPP

#include <mlpack/prereqs.hpp>

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the MiniBatchDiscrimination layer. MiniBatchDiscrimination
 * is a layer of the discriminator that allows the discriminator to look at
 * multiple data examples in combination and perform what is called as
 * mini-batch discrimination.
 * This helps prevent the collapse of the generator parameters to a setting
 * where it emits the same point. This happens because normally a
 * discriminator will process each example independently and there will be
 * no mechanism to diversify the outputs of the generator.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Goodfellow2016,
 *   author  = {Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung,
 *              Alec Radford, Xi Chen},
 *   title   = {Improved Techniques for Training GANs},
 *   year    = {2016},
 *   url     = {https://arxiv.org/abs/1606.03498},
 * }
 * @endcode
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class MiniBatchDiscrimination : public Layer<InputType, OutputType>
{
 public:
  //! Create the MiniBatchDiscrimination object.
  MiniBatchDiscrimination();

  /**
   * Create the MiniBatchDiscrimination layer object using the specified
   * number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param features The number of features to compute for each dimension.
   */
  MiniBatchDiscrimination(const size_t inSize,
                          const size_t outSize,
                          const size_t features);

  /**
   * Reset the layer parameter.
   */
  void Reset();

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed-backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the
   * feed-forward pass.
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
   * @param * (error) The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& /* error */,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the input parameter.
  InputType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  //! Get the gradient.
  OutputType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputType& Gradient() { return gradient; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored dimensions of weight.
  size_t A, B, C;

  //! Locally-stored input batch size.
  size_t batchSize;

  //! Locally-stored temporary features object.
  arma::Mat<typename InputType::elem_type> tempM;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored weight parameters.
  OutputType weight;

  //! Locally-stored features of input.
  arma::Cube<typename InputType::elem_type> M;

  //! Locally-stored delta for features object.
  arma::Cube<typename OutputType::elem_type> deltaM;

  //! Locally-stored L1 distances between features.
  arma::Cube<typename InputType::elem_type> distances;

  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored temporary delta object.
  OutputType deltaTemp;

  //! Locally-stored gradient object.
  OutputType gradient;

  //! Locally-stored input parameter object.
  InputType inputParameter;

  //! Locally-stored output parameter object.
  OutputType outputParameter;
}; // class MiniBatchDiscrimination

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "minibatch_discrimination_impl.hpp"

#endif
