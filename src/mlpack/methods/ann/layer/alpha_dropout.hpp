/**
 * @file methods/ann/layer/alpha_dropout.hpp
 * @author Dakshit Agrawal
 *
 * Definition of the Alpha-Dropout class, which implements a regularizer that
 * randomly sets units to alpha-dash to prevent them from co-adapting and
 * makes an affine transformation so as to keep the mean and variance of
 * outputs at their original values.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_HPP
#define MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * The alpha - dropout layer is a regularizer that randomly with probability
 * 'ratio' sets input values to alphaDash. The alpha - dropout layer is mostly
 * used for SELU activation function where successive layers don't have same
 * mean and variance.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Klambauer2017,
 *   author  = {Gunter Klambauer and Thomas Unterthiner and
 *              Andreas Mayr},
 *   title   = {Self-Normalizing Neural Networks},
 *   journal = {Advances in Neural Information Processing Systems},
 *   year    = {2017},
 *   url     = {https://arxiv.org/abs/1706.02515}
 * }
 * @endcode
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class AlphaDropoutType : public Layer<MatType>
{
 public:
  /**
   * Create the Alpha_Dropout object using the specified ratio.
   *
   * @param ratio The probability of setting a value to alphaDash.
   * @param alphaDash The dropout scaling parameter.
   */
  AlphaDropoutType(const double ratio = 0.5,
                   const double alphaDash = -alpha * lambda);

  /**
   * Clone the AlphaDropoutType object. This handles polymorphism correctly.
   */
  AlphaDropoutType* Clone() const { return new AlphaDropoutType(*this); }

  // Virtual destructor.
  virtual ~AlphaDropoutType() { }

  //! Copy the given AlphaDropoutType layer.
  AlphaDropoutType(const AlphaDropoutType& other);
  //! Take ownership of the given AlphaDropoutType layer.
  AlphaDropoutType(AlphaDropoutType&& other);
  //! Copy the given AlphaDropoutType layer.
  AlphaDropoutType& operator=(const AlphaDropoutType& other);
  //! Take ownership of the given AlphaDropoutType layer.
  AlphaDropoutType& operator=(AlphaDropoutType&& other);

  /**
   * Ordinary feed forward pass of the AlphaDropout layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of the alpha_dropout layer.
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

  //! The probability of setting a value to alphaDash.
  double Ratio() const { return ratio; }

  //! Value to be multiplied with x for affine transformation.
  double A() const { return a; }

  //! Value to be added to a*x for affine transformation.
  double B() const { return b; }

  //! Value of alphaDash.
  double AlphaDash() const { return alphaDash; }

  //! Get the mask.
  const MatType& Mask() const { return mask; }

  //! Modify the probability of setting a value to alphaDash. As
  //! 'a' and 'b' depend on 'ratio', modify them as well.
  void Ratio(const double r)
  {
    ratio = r;
    a = std::pow((1 - ratio) * (1 + ratio * std::pow(alphaDash, 2)), -0.5);
    b = -a * alphaDash * ratio;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored mask object.
  MatType mask;

  //! The probability of setting a value to aplhaDash.
  double ratio;

  //! The low variance value of SELU activation function.
  double alphaDash;

  //! Value of alpha for normalized inputs (taken from SELU).
  static constexpr double alpha = 1.6732632423543772848170429916717;

  //! Value of lambda for normalized inputs (taken from SELU).
  static constexpr double lambda = 1.0507009873554804934193349852946;

  //! Value to be multiplied with x for affine transformation.
  double a;

  //! Value to be added to a*x for affine transformation.
  double b;
}; // class AlphaDropoutType

using AlphaDropout = AlphaDropoutType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "alpha_dropout_impl.hpp"

#endif
