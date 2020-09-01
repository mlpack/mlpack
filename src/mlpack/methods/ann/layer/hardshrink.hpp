/**
 * @file methods/ann/layer/hardshrink.hpp
 * @author Lakshya Ojha
 *
 * Same as soft thresholding, if its amplitude is smaller than a predefined
 * threshold, it will be set to zero (kill), otherwise it will be kept
 * unchanged. In order to promote sparsity and to improve the approximation,
 * the hard thresholding method is used as an alternative.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARDSHRINK_HPP
#define MLPACK_METHODS_ANN_LAYER_HARDSHRINK_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artifical Neural Network. */ {

/**
 * Hard Shrink operator is defined as,
 *
 * \f{eqnarray*}{
 *   f(x) &=& \begin{cases}
 *     x  & : x >  lambda \\
 *     x  & : x < -lambda \\
 *     0  & : otherwise.
 *   \end{cases} \\
 *   f'(x) &=& \begin{cases}
 *     1 & : x >  lambda \\
 *     1 & : x < -lambda \\
 *     0 & : otherwise.
 *   \end{cases}
 * \f}
 *
 * \f$\lambda\f$ is set to 0.5 by default.
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class HardShrink
{
 public:
  /**
   * Create HardShrink object using specified hyperparameter lambda.
   *
   * @param lambda Is calculated by multiplying the
   * 		    noise level sigma of the input(noisy image) and a
   * 		    coefficient 'a' which is one of the training parameters.
   * 		    Default value of lambda is 0.5.
   */
  HardShrink(const double lambda = 0.5);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the Hard Shrink function.
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

  //! Get the hyperparameter lambda.
  double const& Lambda() const { return lambda; }
  //! Modify the hyperparameter lambda.
  double& Lambda() { return lambda; }

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

  //! Locally-stored hyperparameter lambda.
  double lambda;
}; // class HardShrink

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "hardshrink_impl.hpp"

#endif
