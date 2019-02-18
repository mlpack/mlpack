
/**
 * @file contrastive_loss.hpp
 * @author Shardul Shailendra Parab
 *
 * Definition of the contrastive loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CONTRASTIVE_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CONTRASTIVE_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann{

/**
 * The contrastive loss is used to differentiate embeddings
 * using similarity metrics. The feature or latent layer is compared using a similarity metric
 * and trained with the target for a similarity score.
 *
 * For more information refer to the following paper:
 * http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
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
class ContrastiveLoss
{
  public:
    /**
   * Create the ContrastiveLoss object.
   *
   * @param margin 
   * @param lambda - for avoiding poor gradients
   */
    ContrastiveLoss(const double margin = 0.7,
                    const double lambda = pow(10, -6));

    /*
   * Computes the contrastive loss function.
   *
   * @param input1 Input vector 1.
   * @param input1 Input vector 2.
   * @param target The target vector.
   */
    template<typename InputType, typename TargetType>
    double Forward(const InputType&& input1, 
                   const InputType&& input2, const TargetType&& target); 

    /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input1 The propagated input1 (from net 1) activation.
   * @param input2 The propogated input2 (from net 2) activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
    template<typename InputType, typename TargetType, typename OutputType>
    void Backward(const InputType&& input1, 
                const InputType&& input2,
                const TargetType&& target,
                OutputType&& output);

   //! Get the output parameter.
    OutputDataType& OutputParameter() const { return outputParameter; }
   //! Modify the output parameter.
    OutputDataType& OutputParameter() { return outputParameter; }

   //! Get the margin.
    double Margin() const { return margin; }
   //! Modify the margin.
    double& Margin() { return margin; }

   //! Get the lambda.
    double Lambda() const { return lambda; }
   //! Modify the lamda.
    double& Lambda() { return lambda; }

   /**
   * Serialize the layer.
   */
   template<typename Archive>
   void serialize(Archive& ar, const unsigned int /* version */);

  private:
   //! Locally-stored output parameter object.
   OutputDataType outputParameter;

   //! The margin parameter in the loss function
   double margin;

   //! The parameter to avoid poor gradients
   double lambda;
}; // class ContrastiveLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "contrastive_loss_impl.hpp"

#endif
