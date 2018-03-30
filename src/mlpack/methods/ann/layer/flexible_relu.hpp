/**
 * @file flexible_relu.hpp
 * @author Aarush Gupta
 *
 * Definition of FlexibleReLU layer as described by
 * Suo Qiu, Xiangmin Xu and Bolun Cai in
 * "FReLU: Flexible Rectified Linear Units for Improving Convolutional
 *   Neural Networks", 2018
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{
 *  author  = {Suo Qiu, Xiangmin Xu and Bolun Cai},
 *  title   = {FReLU: Flexible Rectified Linear Units for Improving
 *             Convolutional Neural Networks}
 *  journal = {arxiv preprint},
 *  year    = {2018}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_HPP
#define MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /**Artificial Neural Network*/ {

/**
 *The FlexibleReLU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \max(0,x)+alpha \\
 * f'(x) &=& \left\{
 * 	 \begin(array){lr}
 * 	   1 & : x > 0 \\
 * 	   0 & : x \le 0
 * 	 \end{array}
 * \right
 *@f}
 *
 *@tparam InputDataType Type of the input data ( arma::colvec, arma::mar,
 *        arma::sp_mat or arma::cube)
 *
 *@tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *        arma::sp_mat or arma::cube)
 *
 */


template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class FlexibleReLU
{
 public:
  /**
   *
   * Create the FlexibleReLU object using the specified parameters.
   * The non zero parameter can be adjusted by specifying the parameter
   * alpha which controls the range of the relu function. ( Default alpha = 0)
   * This parameter is trainable. 
   *@param alpha Parameter for adjusting the range of the relu function.
   *
   */
  FlexibleReLU(const double alpha = 0);

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
  OutputDataType& Delta() { return delta;}

  //! Get the parameter controlling the range of the relu function.
  double const& Alpha() const { return alpha; }
  //! Modify the parameter controlling the range of the relu function.
  double& Alpha() { return alpha; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version*/);

 private:

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Parameter controlling the range of the rectifier function
  double alpha;
}; // class FlexibleReLU

} // namespace ann
} // namespace mlpack

// Include implementation
#include "flexible_relu_impl.hpp"

#endif
