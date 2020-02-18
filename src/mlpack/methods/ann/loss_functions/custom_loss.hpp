/**
 * @file custom_loss.hpp
 * @author Prince Gupta
 *
 * Definition for custom loss function class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_CUSTOM_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_CUSTOM_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * We can define any custom loss function after defining how it works
 * on basic level.
 * 
 * @tparam InputDataType Type of input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class CustomLoss
{
 public:
  /**
   * Create the custom loss function with defined forward and backward
   * propagation functions.
   * 
   * @param forward custom forward function for the loss that takes in
   *        input and target matrices as parameter and returns loss.
   * @param backward custom backward function for the loss that takes in
   *        input, target and output matrices as parameter and saves the
   *        gradient in the ouput matrix.
   */
  CustomLoss(
      std::function<double(const InputDataType&&,
                           const InputDataType&&)> forward,
      std::function<void(const InputDataType&&,
                         const InputDataType&&, OutputDataType&&)> backward);
  /**
   * Computes the specified custom loss function.
   * 
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& input, const TargetType&& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType&& input,
                const TargetType&& target,
                OutputType&& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
  //! Locally-stored forward function.
  std::function<double(const InputDataType&&, const InputDataType&&)> forward;
  //! Locally-stored backward function.
  std::function<void(const InputDataType&&,
                     const InputDataType&&,
                     OutputDataType&&)> backward;
}; // class CustomLoss

} // namespace ann
} // namespace mlpack

// Include implementation
#include "custom_loss_impl.hpp"

#endif
