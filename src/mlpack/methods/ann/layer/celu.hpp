/**
 * @file methods/ann/layer/celu.hpp
 * @author Gaurav Singh
 *
 * Definition of the CELU activation function as described by Jonathan T. Barron.
 *
 * For more information, read the following paper.
 *
 * @code
 * @article{
 *   author  = {Jonathan T. Barron},
 *   title   = {Continuously Differentiable Exponential Linear Units},
 *   year    = {2017},
 *   url     = {https://arxiv.org/pdf/1704.07483}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CELU_HPP
#define MLPACK_METHODS_ANN_LAYER_CELU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The CELU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *    x & : x \ge 0 \\
 *    \alpha(e^(\frac{x}{\alpha}) - 1) & : x < 0
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x \ge 0 \\
 *     (\frac{f(x)}{\alpha}) + 1 & : x < 0
 *   \end{array}
 * \right.
 * @f}
 *
 * In the deterministic mode, there is no computation of the derivative.
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
class CELU
{
 public:
  /**
   * Create the CELU object using the specified parameter. The non zero
   * gradient for negative inputs can be adjusted by specifying the CELU
   * hyperparameter alpha (alpha > 0).
   *
   * @param alpha Scale parameter for the negative factor (default = 1.0).
   */
  CELU(const double alpha = 1.0);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation f(x).
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& input, const DataType& gy, DataType& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the non zero gradient.
  double const& Alpha() const { return alpha; }
  //! Modify the non zero gradient.
  double& Alpha() { return alpha; }

  //! Get the value of deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of deterministic parameter.
  bool& Deterministic() { return deterministic; }

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

  //! Locally stored first derivative of the activation function.
  arma::mat derivative;

  //! CELU Hyperparameter (alpha > 0).
  double alpha;

  //! If true the derivative computation is disabled, see notes above.
  bool deterministic;
}; // class CELU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "celu_impl.hpp"

#endif
