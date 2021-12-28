/**
 * @file methods/ann/layer/isru.hpp
 * @author Suvarsha Chennareddy
 *
 * Definition of the ISRU activation function as described by Jonathan T. Barron.
 *
 * For more information, read the following paper (page 6).
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
#ifndef MLPACK_METHODS_ANN_LAYER_ISRU_HPP
#define MLPACK_METHODS_ANN_LAYER_ISRU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The ISRU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \frac{x}{\sqrt{1 + \alpha x^2}} \\
 * f'(x) &=& \frac{1}{\sqrt{1 + \alpha x^2}^3} \\
 * @f}
 *
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<typename InputDataType = arma::mat,
         typename OutputDataType = arma::mat>
class ISRU
{
 public:

 /**
   * Create the ISRU object using the specified parameter.
   *
   * @param alpha Positive scale parameter.
  */
  ISRU(const double alpha = 1.0);

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
  
    //! Get the positive scale parameter.
  double const& Alpha() const { return alpha; }
  
  //! Modify the positive scale parameter.
  double& Alpha() { return alpha; }

  //! Get size of weights.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! ISRU Hyperparameter (alpha > 0).
  double alpha;
}; // class ISRU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "isru_impl.hpp"

#endif
