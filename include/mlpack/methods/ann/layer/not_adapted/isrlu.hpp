/**
 * @file methods/ann/layer/isrlu.hpp
 * @author Abhinav Anand
 *
 * Definition of the ISRLU activation function as described by Jonathan T. Barron.
 *
 * For more information, read the following paper.
 *
 * @code
 * @article{
 *   author  = {Carlile, Brad and Delamarter, Guy and Kinney, Paul and Marti,
 *              Akiko and Whitney, Brian},
 *   title   = {Improving deep learning by inverse square root linear units (ISRLUs)},
 *   year    = {2017},
 *   url     = {https://arxiv.org/pdf/1710.09967.pdf}
 * }
 * @endcode
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

/**
 * The ISRLU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *    x & : x \ge 0 \\
 *    x(\frac{1}{1 + \alpha x^2}) & : x < 0
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *    x & : 1 \ge 0 \\
 *    (\frac{1}{1 + \alpha x^2})^3 & : x < 0
 *   \end{array}
 * \right. 
 * @f}
 *
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
   * Create the ISRLU object using the specified parameter.
   *
   * @param alpha Scale parameter controls the value to which an ISRLU
   *        saturates for negative inputs.
   */
  ISRLU(const double alpha = 1.0);

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

  //! Get size of weights.
  size_t WeightSize() { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally stored first derivative of the activation function.
  arma::mat derivative;

  //! ISRLU Hyperparameter (alpha > 0).
  double alpha;
}; // class ISRLU

} // namespace mlpack

// Include implementation.
#include "isrlu_impl.hpp"

#endif
