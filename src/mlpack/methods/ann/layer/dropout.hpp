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

/**
 * The dropout layer is a regularizer that randomly with probability 'ratio'
 * sets input values to zero and scales the remaining elements by factor 1 /
 * (1 - ratio) rather than during test time so as to keep the expected sum same.
 * When the layer is in testing mode, there is no change in the input.
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
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class DropoutType : public Layer<MatType>
{
 public:
  /**
   * Create the Dropout object using the specified ratio parameter.
   *
   * @param ratio The probability of setting a value to zero.
   */
  DropoutType(const double ratio = 0.5);

  //! Clone the DropoutType object. This handles polymorphism correctly.
  DropoutType* Clone() const { return new DropoutType(*this); }

  // Virtual destructor.
  virtual ~DropoutType() { }

  //! Copy the given DropoutType.
  DropoutType(const DropoutType& other);
  //! Take ownership of the given DropoutType.
  DropoutType(DropoutType&& other);
  //! Copy the given DropoutType.
  DropoutType& operator=(const DropoutType& other);
  //! Take ownership of the given DropoutType.
  DropoutType& operator=(DropoutType&& other);

  /**
   * Ordinary feed forward pass of the dropout layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of the dropout layer.
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
  //! Locally-stored mask object.
  MatType mask;

  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;
}; // class DropoutType

// Convenience typedefs.

// Standard Dropout layer.
using Dropout = DropoutType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "dropout_impl.hpp"

#endif
