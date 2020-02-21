/**
 * @file isrlu.hpp
 * @author Pranav Reddy P
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ISRLU_HPP
#define MLPACK_METHODS_ANN_LAYER_ISRLU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The ISRLU function, also known as Inverse Square Root Linear Unit
 * It has a negative value allowing it to push the mean unit activation closer
 * to zero and bring the normal gradient closer to the unit natural gradient,
 * ensuring a noise robust deactivation state, lessening the over fitting
 * risk. Experiments have led to the conclusion that ISRLU leads to faster
 * learning and better generalization on CNN's than ReLU.
 *
 * It is defined by
 * 
 * @f{eqnarray*}{
 * f(x) &=& x/sqrt(1 + alpha*x*x) if x<0
 *      &=& x                     if x>=0
 * f'(x) &=& cube(1/sqrt(1 + alpha*x*x)) if x<0
 *       &=& 1                           if x>=0
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
class ISRLU
{
 public:
  /**
   * Create the ISRLU object using the specified parameters.
   * The non zero gradient can be adjusted by specifying the parameter
   * alpha in the range 0 to 1. Default value of alpha = 0.03
   *
   * @param alpha Non zero gradient.
   */
  ISRLU(const double alpha = 0.03);

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

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the Hyperparameter Alpha.
  double const& Alpha() const { return alpha; }
  //! Modify the non zero gradient.
  double& Alpha() { return alpha; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   * Computes the ISRLU function.
   *
   * @param x Input data.
   * @return f(x).
   */
  double Fn(const double x)
  {
    if (alpha <= -1 / std::pow(x, 2))
      return DBL_MAX;
    if (x < 0)
      return x / (std::sqrt(1 + alpha * std::pow(2, x)));
    else
      return x;
  }

  /**
   * Computes the ISRLU function using a dense matrix as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    if (alpha <= -1 / std::pow(x, 2))
      y = DBL_MAX;
    if (x < 0)
      y = x / (arma::sqrt(1 + alpha * arma::pow(x, 2)));
    else
      y = x;
  }

  /**
   * Computes the first derivative of the ISRLU function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  double Deriv(const double x)
  {
    if (alpha <= -1 / (std::pow(x, 2)))
      return DBL_MAX;
    return (x >= 0) ? 1 : std::pow
    (1 / std::sqrt(1 + alpha * std::pow(x, 2)), 3);
  }

  /**
   * Computes the first derivative of the ISRLU function.
   *
   * @param x Input activations.
   * @param y The resulting derivatives.
   */

  template<typename InputType, typename OutputType>
  void Deriv(const InputType& x, OutputType& y)
  {
    y.set_size(arma::size(x));

    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = Deriv(x(i));
    }
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! ISRLU Hyperparameter (0 < alpha)
  double alpha;
}; // class ISRLU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "isrlu_impl.hpp"

#endif
