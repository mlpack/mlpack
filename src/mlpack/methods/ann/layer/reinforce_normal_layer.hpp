/**
 * @file reinforce_normal_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the ReinforceNormalLayer class, which implements the REINFORCE
 * algorithm for the normal distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_REINFORCE_NORMAL_LAYER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the reinforce normal layer. The reinforce normal layer
 * implements the REINFORCE algorithm for the normal distribution.
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
class ReinforceNormalLayer
{
 public:
  /**
   * Create the ReinforceNormalLayer object.
   *
   * @param stdev Standard deviation used during the forward and backward pass.
   */
  ReinforceNormalLayer(const double stdev) : stdev(stdev)
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    if (!deterministic)
    {
      // Multiply by standard deviations and re-center the means to the mean.
      output = arma::randn<arma::Mat<eT> >(input.n_rows, input.n_cols) *
          stdev + input;
    }
    else
    {
      // Use maximum a posteriori.
      output = input;
    }
  }

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
  void Backward(const DataType& input,
                const DataType& /* gy */,
                DataType& g)
  {
    g = (input - inputParameter) / std::pow(stdev, 2.0);

    // Multiply by reward and multiply by -1.
    g *= -reward;
  }


  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the value of the reward parameter.
  double Reward() const { return reward; }
  //! Modify the value of the deterministic parameter.
  double& Reward() { return reward; }

 private:
  //! Standard deviation used during the forward and backward pass.
  const double stdev;

  //! Locally-stored reward parameter.
  double reward;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! If true use maximum a posteriori during the forward pass.
  bool deterministic;
}; // class ReinforceNormalLayer

}; // namespace ann
}; // namespace mlpack

#endif
