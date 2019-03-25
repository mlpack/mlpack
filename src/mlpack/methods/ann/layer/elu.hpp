/**
 * @file elu.hpp
 * @author Vivek Pal
 * @author Dakshit Agrawal
 *
 * Definition of the ELU activation function as descibed by Djork-Arne Clevert,
 * Thomas Unterthiner and Sepp Hochreiter.
 *
 * Definition of the SELU function as introduced by
 * Klambauer et. al. in Self Neural Networks.  The SELU activation
 * function keeps the mean and variance of the input invariant.
 *
 * In short, SELU = lambda * ELU, with 'alpha' and 'lambda' fixed for
 * normalized inputs.
 *
 * Hence both ELU and SELU are implemented in the same file, with
 * lambda = 1 for ELU function.
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
 *    \alpha(e^x - 1) & : x \le 0
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     f(x) + \alpha & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{Clevert2015,
 *   author  = {Djork{-}Arn{\'{e}} Clevert and Thomas Unterthiner and
 *              Sepp Hochreiter},
 *   title   = {Fast and Accurate Deep Network Learning by Exponential Linear
 *              Units (ELUs)},
 *   journal = {CoRR},
 *   year    = {2015}
 * }
 * @endcode
 *
 *
 * The SELU activation function is defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *    \lambda * x & : x > 0 \\
 *    \lambda * \alpha(e^x - 1) & : x \le 0
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     \lambda & : x > 0 \\
 *     f(x) + \lambda * \alpha & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{Klambauer2017,
 *   author  = {Gunter Klambauer and Thomas Unterthiner and
 *              Andreas Mayr},
 *   title   = {Self-Normalizing Neural Networks},
 *   journal = {Advances in Neural Information Processing Systems},
 *   year    = {2017}
 * }
 * @endcode
 *
 * In the deterministic mode, there is no computation of the derivative.
 *
 * @note During training deterministic should be set to false and during
 *       testing/inference deterministic should be set to true.
 * @note Make sure to use SELU activation function with normalized inputs and
 *       weights initialized with Lecun Normal Initialization.
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
   * Create the ELU object.
   *
   * NOTE: Use this constructor for SELU activation function.
   */
  ELU();

  /**
   * Create the ELU object using the specified parameter. The non zero
   * gradient for negative inputs can be adjusted by specifying the ELU
   * hyperparameter alpha (alpha > 0).
   *
   * @note Use this constructor for ELU activation function.
   * @param alpha Scale parameter for the negative factor.
   */
  ELU(const double alpha);

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
   * @param input The propagated input activation f(x).
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType&& input, DataType&& gy, DataType&& g);

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

  //! Get the lambda parameter.
  double const& Lambda() const { return lambda; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * Computes the value of activation function.
   *
   * @param x Input data.
   * @return f(x).
   */
  double Fn(const double x)
  {
    if (x < DBL_MAX)
    {
      return (x > 0) ? lambda * x : lambda * alpha * (std::exp(x) - 1);
    }

    return 1.0;
  }

  /**
   * Computes the value of activation function using a dense matrix as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = Fn(x(i));
    }
  }

  /**
   * Computes the first derivative of the activation function.
   *
   * @param x Input data.
   * @param y Propagated data f(x).
   * @return f'(x)
   */
  double Deriv(const double x, const double y)
  {
    return (x > 0) ? lambda : y + lambda * alpha;
  }

  /**
   * Computes the first derivative of the activation function.
   *
   * @param x Input data.
   * @param y Output activations f(x).
   * @param z The resulting derivatives.
   */
  template<typename InputType, typename OutputType>
  void Deriv(const InputType& x, OutputType& y)
  {
    derivative.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; i++)
    {
      derivative(i) = Deriv(x(i), y(i));
    }
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally stored first derivative of the activation function.
  arma::mat derivative;

  //! ELU Hyperparameter (0 < alpha)
  //! SELU parameter fixed to 1.6732632423543774 for normalized inputs.
  double alpha;

  //! Lambda Parameter used for multiplication of ELU function.
  //! For ELU activation function, lambda = 1.
  //! For SELU activation function, lambda = 1.0507009873554802 for normalized
  //! inputs.
  double lambda;

  //! If true the derivative computation is disabled, see notes above.
  bool deterministic;
}; // class ELU

// Template alias for SELU using ELU class.
using SELU = ELU<arma::mat, arma::mat>;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "elu_impl.hpp"

#endif
