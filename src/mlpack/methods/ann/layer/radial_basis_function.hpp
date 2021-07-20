/**
 * @file radial_basis_function.hpp
 * @author Himanshu Pathak
 *
 * Definition of the Radial Basis Function module class.
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RBF_HPP
#define MLPACK_METHODS_ANN_LAYER_RBF_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/activation_functions/gaussian_function.hpp>

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


/**
 * Implementation of the Radial Basis Function layer. The RBF class when use
 * with a non-linear activation function acts as a Radial Basis Function which
 * can be used with Feed-Forward neural network.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{Volume 51: Artificial Intelligence and Statistics,
 *   author  = {Qichao Que, Mikhail Belkin},
 *   title   = {Back to the Future: Radial Basis Function Networks Revisited},
 *   year    = {2016},
 *   url     = {http://proceedings.mlr.press/v51/que16.pdf},
 * }
 * @endcode
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam Activation Type of the activation function (mlpack::ann::Gaussian).
 */

template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat,
    typename Activation = GaussianFunction
>
class RBF : public Layer<InputType, OutputType>
{
 public:
  //! Create the RBF object.
  RBF();

  /**
   * Create the Radial Basis Function layer object using the specified
   * parameters.
   *
   * @param outSize The number of output units.
   * @param centres The centres calculated using k-means of data.
   * @param betas The beta value to be used with centres.
   */
  RBF(const size_t outSize,
      InputType& centres,
      double betas = 0);

  /**
   * Ordinary feed forward pass of the radial basis function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of the radial basis function.
   */
  void Backward(const InputType& /* input */,
                const OutputType& /* gy */,
                OutputType& /* g */);

  const std::vector<size_t>& OutputDimensions() const
  {
    std::vector<size_t> outputDimensions(inputDimensions.size(), 1);
    // This flattens the input.
    outputDimensions[0] = outSize;
    return outputDimensions;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored the betas values.
  double betas;

  //! Locally-stored the learnable centre of the shape.
  InputType centres;

  //! Locally-stored the output distances of the shape.
  OutputType distances;
}; // class RBF

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "radial_basis_function_impl.hpp"

#endif
