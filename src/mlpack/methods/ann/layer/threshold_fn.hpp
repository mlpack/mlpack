/**
 * @file methods/ann/layer/threshold_fn.hpp
 * @author Shubham Agrawal
 *
 * Definition of the Threshold activation function
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_THRESHOLD_HPP
#define MLPACK_METHODS_ANN_LAYER_THRESHOLD_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Threshold activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *     x & : x > threshold \\
 *     value & : x \le threshold
 *   \end{array}
 * \right.
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
class Threshold
{
 public:
  /**
   * Create the Threshold object using the specified parameters.
   * The non zero gradient region can be adjusted by specifying the parameter
   * threshold. Default (threshold = 0.0)
   * The value when x < threshold can be adjusted by specifying the parameter
   * value. Default (value = 0.0)
   *
   * @param threshold Non zero gradient region threshold
   * @param value Value when x < threshold
   */
  Threshold(const double threshold = 0.0,
            const double value = 0.0);

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
   * @param input The propagated input activation.
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

  //! Get the non zero gradient region threshold.
  double const& ThresholdVar() const { return threshold; }
  //! Modify the non zero gradient region threshold.
  double& ThresholdVar() { return threshold; }

  //! Get the zero gradient region value.
  double const& Value() const { return value; }
  //! Modify the zero gradient region value.
  double& Value() { return value; }

  //! Get size of weights.
  size_t WeightSize() const { return 0; }

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

  //! Locally-stored +ve output indexes object.
  arma::uvec positive;

  //! Threshold Parameter
  double threshold;

  //! Value Parameter
  double value;
}; // class Threshold

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "threshold_fn_impl.hpp"

#endif
