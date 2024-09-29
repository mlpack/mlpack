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
   * @param outSize The number of output units.
   * @param features The number of features to compute for each dimension.
   */
  MiniBatchDiscrimination(const size_t outSize,
                          const size_t features);

  /**
   * Reset the layer parameter.
   */
  void SetWeights(typename OutputType::elem_type* weightsPtr);

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

  const size_t WeightSize() const { return a * b * c; }

  const std::vector<size_t> OutputDimensions() const
  {
    a = std::accumulate(inputDimensions.begin(), inputDimensions.end(), 0);
    std::vector<size_t> outputDimensions(inputDimensions.size(), 1);
    // TODO: not sure if this is right... we just interpret it all as
    // one-dimensional.
    outputDimensions[0] = a + b;

    return outputDimensions;
  }

  //! Get the shape of the input.
  size_t InputShape() const
  {
    return A;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored dimensions of weight.
  size_t a, b, c;

  //! Locally-stored input batch size.
  size_t batchSize;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored features of input.  Cached to avoid recomputation.
  InputType M;

  //! Locally-stored delta for features object.
  arma::Cube<typename OutputType::elem_type> deltaM;

  //! Locally-stored L1 distances between features.
  arma::Cube<typename InputType::elem_type> distances;
}; // class MiniBatchDiscrimination

} // namespace mlpack

// Include implementation.
#include "minibatch_discrimination_impl.hpp"

#endif
