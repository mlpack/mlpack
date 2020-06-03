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
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<typename InputDataType = arma::mat,
         typename OutputDataType = arma::mat>
class Dropout
{
 public:
  /**
   * Create the Dropout object using the specified ratio parameter.
   *
   * @param ratio The probability of setting a value to zero.
   */
  Dropout(const double ratio = 0.5);

  /**
   * Ordinary feed forward pass of the dropout layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of the dropout layer.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the detla.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
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
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored mast object.
  OutputDataType mask;

  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;
}; // class Dropout

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "dropout_impl.hpp"

#endif
