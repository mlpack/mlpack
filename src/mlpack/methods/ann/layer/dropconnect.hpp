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

/**
 * The DropConnect layer is a regularizer that randomly with probability
 * ratio sets the connection values to zero and scales the remaining
 * elements by factor 1 /(1 - ratio). The output is scaled with 1 / (1 - p)
 * when in training mode.  During testing, the layer just computes the output.
 * The output is computed according to the input layer. If no input layer is
 * given, it will take a linear layer as default.
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
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class DropConnectType : public Layer<MatType>
{
 public:
  //! Create the DropConnect object.
  DropConnectType();

  /**
   * Creates the DropConnect Layer as a Linear Object that takes the number of
   * output units and a ratio as parameter.
   *
   * @param outSize The number of output units.
   * @param ratio The probability of setting a value to zero.
   */
  DropConnectType(const size_t outSize,
                  const double ratio = 0.5);

  //! Clone the DropConnectType object. This handles polymorphism correctly.
  DropConnectType* Clone() const { return new DropConnectType(*this); }

  // Virtual destructor.
  virtual ~DropConnectType();

  //! Copy the given DropConnectType (except for weights).
  DropConnectType(const DropConnectType& other);
  //! Take ownership of the given DropConnectType (except for weights).
  DropConnectType(DropConnectType&& other);
  //! Copy the given DropConnectType (except for weights).
  DropConnectType& operator=(const DropConnectType& other);
  //! Take ownership of the given DropConnectType (except for weights).
  DropConnectType& operator=(DropConnectType&& other);

  /**
   * Ordinary feed forward pass of the DropConnect layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of the DropConnect layer.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The propagated input.
   * @param error The calculated error.
   * @param * (gradient) The calculated gradient.
   */
  void Gradient(const MatType& input, const MatType& error, MatType& gradient);

  //! The probability of setting a value to zero.
  double Ratio() const { return ratio; }

  //! Modify the probability of setting a value to zero.
  void Ratio(const double r)
  {
    ratio = r;
    scale = 1.0 / (1.0 - ratio);
  }

  //! Compute the output dimensions of the layer based on `InputDimensions()`.
  void ComputeOutputDimensions();

  //! Return the size of the weights.
  size_t WeightSize() const { return baseLayer->WeightSize(); }

  // Set the weights to use the given memory `weightsPtr`.
  void SetWeights(const MatType& weightsIn);

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

  //! Locally-stored mask object.
  MatType mask;

  //! Denoise mask for the weights.
  MatType denoise;

  //! Locally-stored layer module.
  Layer<MatType>* baseLayer;
}; // class DropConnect.

// Convenience typedefs.

// Standard DropConnect layer.
using DropConnect = DropConnectType<arma::mat>;

}  // namespace mlpack

// Include implementation.
#include "dropconnect_impl.hpp"

#endif
