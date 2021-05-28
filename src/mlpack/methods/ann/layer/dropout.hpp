/**
 * @file methods/ann/layer/dropout.hpp
 * @author Marcus Edel
 *
 * Definition of the Dropout class, which implements a regularizer that
 * randomly sets units to zero preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPOUT_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPOUT_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The dropout layer is a regularizer that randomly with probability 'ratio'
 * sets input values to zero and scales the remaining elements by factor 1 /
 * (1 - ratio) rather than during test time so as to keep the expected sum same.
 * In the deterministic mode (during testing), there is no change in the input.
 *
 * Note: During training you should set deterministic to false and during
 * testing you should set deterministic to true.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Hinton2012,
 *   author  = {Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky,
 *              Ilya Sutskever, Ruslan Salakhutdinov},
 *   title   = {Improving neural networks by preventing co-adaptation of feature
 *              detectors},
 *   journal = {CoRR},
 *   volume  = {abs/1207.0580},
 *   year    = {2012},
 *   url     = {https://arxiv.org/abs/1207.0580}
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
class DropoutType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the Dropout object using the specified ratio parameter.
   *
   * @param ratio The probability of setting a value to zero.
   */
  DropoutType(const double ratio = 0.5);

  //! Copy Constructor.
  DropoutType(const DropoutType& layer);

  //! Move Constructor.
  DropoutType(const DropoutType&&);

  //! Copy assignment operator.
  DropoutType& operator=(const DropoutType& layer);

  //! Move assignment operator.
  DropoutType& operator=(DropoutType&& layer);

  //! Clone the DropoutType object. This handles polymorphism correctly.
  DropoutType* Clone() const { return new DropoutType(*this); }

  /**
   * Ordinary feed forward pass of the dropout layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of the dropout layer.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

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

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored mast object.
  OutputType mask;

  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;
}; // class DropoutType

// Convenience typedefs.

// Standard Dropout layer.
typedef DropoutType<arma::mat, arma::mat> Dropout;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "dropout_impl.hpp"

#endif
