/**
 * @file elu.hpp
 * @author Vivek Pal
 *
 * Definition of the ELU activation function as descibed by Djork-Arne Clevert,
 * Thomas Unterthiner & Sepp Hochreiter.
 *
 * For more information, read the following paper:
 * 
 * @code
 * @conference{ICLR2016,
 *   author = {Djork-Arne Clevert, Thomas Unterthiner & Sepp Hochreiter},
 *   title = {Fast and Accurate Deep Network Learning by Exponential Linear
 *   Units (ELUs},
 *   year = {2015}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ELU_HPP
#define MLPACK_METHODS_ANN_LAYER_ELU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The ELU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *    x & : x > 0 \\
 *    alpha(e^x - 1) & : x \le 0
 *   \end{array}
 * \right
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     y + alpha & : x \le 0
 *   \end{array}
 * \right
 * @f}
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
class ELU
{
 public:
  /**
   * Create the ELU object using the specified parameters. The non zero
   * gradient for negative inputs can be adjusted by specifying the ELU
   * hyperparameter alpha (alpha > 0). Default alpha = 1.0
   *
   * @param alpha Non zero gradient
   */
  ELU(const double alpha = 1.0);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType&& input, OutputType&& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType&& input, DataType&& gy, DataType&& g);

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

  //! Get the non zero gradient.
  double const& Alpha() const { return alpha; }
  //! Modify the non zero gradient.
  double& Alpha() { return alpha; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * Computes the ELU function
   *
   * @param x Input data.
   * @return f(x).
   */
  double fn(const double x)
  {
    if (x < DBL_MAX)
      return (x > 0) ? x : alpha * (std::exp(x) - 1);
    return 1.0;
  }

  /**
   * Computes the ELU function using a dense matrix as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y = x;

    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = fn(x(i));
    }
  }

  /**
   * Computes the first derivative of the ELU function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  double Deriv(const double y)
  {
    return (y > 0) ? 1 : (y + alpha);
  }

  /**
   * Computes the first derivative of the ELU function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */

  template<typename InputType, typename OutputType>
  void Deriv(const InputType& x, OutputType& y)
  {
    y = x;

    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = Deriv(x(i));
    }
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! ELU Hyperparameter (0 < alpha)
  double alpha;

}; // class ELU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "elu_impl.hpp"

#endif