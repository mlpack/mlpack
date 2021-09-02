/**
 * @file methods/ann/layer/sqnl.hpp
 * @author Shaikh Yusuf Niaz
 *
 * Definition of Square NonLinearity (SQNL) function as described by
 * Wuraola, Adedamola and Patel, Nitish.
 *
 * For more information, read the following paper.
 *
 * @code
 * @INPROCEEDINGS{8489043,
 *  author={Wuraola, Adedamola and Patel, Nitish},
 *  booktitle={2018 International Joint Conference on Neural Networks (IJCNN)}, 
 *  title={SQNL: A New Computationally Efficient Activation Function}, 
 *  year={2018},
 *  volume={},
 *  number={},
 *  pages={1-7},
 *  doi={10.1109/IJCNN.2018.8489043}}
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SQNL_HPP
#define MLPACK_METHODS_ANN_LAYER_SQNL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Square NonLinearity function, defined by
 *
 * @f{eqnarray*}{
 *   f(x) &=& \begin{cases}
 *     1 & x>2.0\\
 *     x - \fraq{x^2}{4} & 0\leq x\leq 2.0\\
 *     x + \fraq{x^2}{4} & -2.0\leq x<0\\
 *    -1 & x<-2.0\\
 *   \end{cases} \\
 *   f'(x) &=& \begin{cases}
 *     0 & x>2.0\\
 *     1 - \fraq{x}{2} & 0\leq x\leq 2.0\\
 *     1 + \fraq{x}{2} & -2.0\leq x<0\\
 *     0 & x<-2.0\\
 *   \end{cases}
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
class SQNL
{
 public:
  /**
   * Create the SQNL object 
   */
  SQNL();

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

  //! Get size of weights.
  size_t WeightSize() { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally stored first derivative of the activation function.
  arma::mat derivative;

}; // class SQNL

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "sqnl_impl.hpp"

#endif
