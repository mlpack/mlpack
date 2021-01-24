/**
 * @file methods/ann/layer/dropconnect.hpp
 * @author Palash Ahuja
 * @author Marcus Edel
 *
 * Definition of the DropConnect class, which implements a regularizer
 * that randomly sets connections to zero. Preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPCONNECT_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPCONNECT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The DropConnect layer is a regularizer that randomly with probability
 * ratio sets the connection values to zero and scales the remaining
 * elements by factor 1 /(1 - ratio). The output is scaled with 1 / (1 - p)
 * when deterministic is false. In the deterministic mode(during testing),
 * the layer just computes the output. The output is computed according
 * to the input layer. If no input layer is given, it will take a linear layer
 * as default.
 *
 * Note:
 * During training you should set deterministic to false and during testing
 * you should set deterministic to true.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{WanICML2013,
 *   title     = {Regularization of Neural Networks using DropConnect},
 *   booktitle = {Proceedings of the 30th International Conference on Machine
 *                Learning(ICML - 13)},
 *   author    = {Li Wan and Matthew Zeiler and Sixin Zhang and Yann L. Cun and
 *                Rob Fergus},
 *   year      = {2013},
 *   url       = {http://proceedings.mlr.press/v28/wan13.pdf}
 * }
 * @endcode
 *
 * @tparam InputType The type of the layer's inputs. The layer automatically
 *     cast inputs to this type (Default: arma::mat).
 * @tparam OutputType The type of the computation which also causes the output
 *     to also be in this type. The type also allows the computation and weight
 *     type to differ from the input type (Default: arma::mat).
 */
template<typename InputType = arma::mat, typename OutputType = arma::mat>
class DropConnectType : public Layer<InputType, OutputType>
{
 public:
  //! Create the DropConnect object.
  DropConnectType();

  /**
   * Creates the DropConnect Layer as a Linear Object that takes input size,
   * output size and ratio as parameter.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param ratio The probability of setting a value to zero.
   */
  DropConnectType(const size_t inSize,
                  const size_t outSize,
                  const double ratio = 0.5);

  /**
   * Ordinary feed forward pass of the DropConnect layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of the DropConnect layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input, const OutputType& gy, OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The propagated input.
   * @param error The calculated error.
   * @param * (gradient) The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the model modules.
  std::vector<Layer<InputType, OutputType>*>& Model() { return network; }

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! The value of the deterministic parameter.
  bool const& Deterministic() const { return deterministic; }

  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! The probability of setting a value to zero.
  double Ratio() const { return ratio; }

  //! Modify the probability of setting a value to zero.
  void Ratio(const double r)
  {
    ratio = r;
    scale = 1.0 / (1.0 - ratio);
  }

  //! Return the size of the weight matrix.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;

  //! Locally-stored weight object.
  OutputType weights;

  //! Locally-stored mask object.
  OutputType mask;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! Denoise mask for the weights.
  OutputType denoise;

  //! Locally-stored layer module.
  Layer<InputType, OutputType>* baseLayer;

  //! Locally-stored network modules.
  std::vector<Layer<InputType, OutputType>*> network;
}; // class DropConnect.

// Convenience typedefs.

// Standard DropConnect layer.
typedef DropConnectType<arma::mat, arma::mat> DropConnect;

}  // namespace ann
}  // namespace mlpack

// Include implementation.
#include "dropconnect_impl.hpp"

#endif
