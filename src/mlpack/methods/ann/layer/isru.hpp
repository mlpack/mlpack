/**
 * @file isru.hpp
 * @author Prince Gupta
 *
 * Definition of ISRU (Inverse Square Root Unit) activation function.
 *
 * For more information, refer the following paper.
 *
 * @code
 * @article{Carlile2017,
 *   author  = {Brad Carlile, Guy Delamarter, Paul Kinney, Akiko Marti,
 *              Brian Whitney},
 *   title   = {Improving Deep Learning by Inverse Square Root Linear
 *              Units (ISRLUs)},
 *   year    = {2017},
 *   url     = {https://arxiv.org/abs/1710.09967},
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
namespace ann /** Artifical Neural Network. */ {

/**
 * The ISRU (Inverse Square Root Unit) activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& x / sqrt(1 + alpha* x ^ {2})
 * f'(x) &=& cube(1 / sqrt(1 + alpha * x ^ {2}))
 * @f
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class ISRU
{
 public:
  /**
   * Create ISRU object using specified Hyperparameter alpha.
   * Hyperparameter alpha controls the value to which the ISRU function
   * saturates (default = 0.1).
   * Alpha should be > 0.
   *
   * @param alpha hyperparameter used to calculate ISRU function.
   */
  ISRU(const double alpha = 0.1);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   * 
   * @param input Input data used for evaluating the ISRU function.
   * @param output Resulting output activation
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
  void Backward(const DataType& input,
                DataType& gy,
                DataType& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the hyperparameter alpha.
  double const& Alpha() const { return alpha; }
  //! Modify the hyperparameter alpha.
  double& Alpha() { return alpha; }

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

  //! ISRU hyperparameter.
  double alpha;
}; // class ISRU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "isru_impl.hpp"

#endif
