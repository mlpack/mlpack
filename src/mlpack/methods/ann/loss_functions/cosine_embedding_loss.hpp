/**
 * @file cosine_embedding_loss.hpp
 * @author Kartik Dutt
 *
 * Definition of the Cosine Embedding loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_COSINE_EMBEDDING_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_COSINE_EMBEDDING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Cosine Embeddings Loss function is used for measuring whether two inputs are
 * similar or dissimilar, using using the cosine distance, and is typically used
 * for learning nonlinear embeddings or semi-supervised learning.
 *
 * @f{eqnarray*}{
 * f(x) = 1 - cos(x1, x2) , for y = 1
 * f(x) = max(0, cos(x1, x2) - margin) , for y = -1
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
class CosineEmbeddingLoss
{
 public:
  /**
   * Create the CosineEmbeddingLoss object.
   * 
   * @param margin Increases cosine distance in case of dissimilarity.
   *               Refer definition of cosine-embedding-loss above.
   * @param takeMean Boolean variable to specify whether to take mean or not.
   *                 Specifies reduction method i.e. sum or mean corresponding
   *                 to 0 and 1 respectively. Default value = 0.
   */
  CosineEmbeddingLoss(const double margin = 0.0, const bool takeMean = false);

  /**
   * Ordinary feed forward pass of a neural network.
   *
   * @param x1 Input data used for evaluating the given function.
   * @param x2 Input data used for evaluating the given function.
   * @param y Input data used to determine whether to calculate
   *          cosine similarity or dissimilarity. It should only
   *          contain values equal to 1 or -1.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& x1, const InputType x2,
      const TargetType&& y);

  /**
   * Ordinary feed backward pass of a neural network. The negative log
   * likelihood layer expects that the input contains log-probabilities for
   * each class. The layer also expects a class index, in the range between 1
   * and the number of classes, as target when calling the Forward function.
   *
   * @param input The propagated input activation.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType&& input,
                const TargetType&& target,
                OutputType&& output);

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

  //! Get the value of takeMean.
  bool TakeMean() const { return takeMean; }
  //! Modify the value of takeMean.
  bool& TakeMean() { return takeMean; }

  //! Get the value of margin.
  bool Margin() const { return margin; }
  //! Modify the value of takeMean.
  bool& Margin() { return margin; }
  

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  // Returns Cosine-Distance between two vectors.
  template <typename InputType>
  double CosineDistance(InputType x1, InputType x2)
  {
    return 1.0 - arma::accu(x1 % x2) /
        std::sqrt(arma::accu(arma::pow(x1, 2)) * arma::accu(arma::pow(x2, 2)));
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored value of margin parameter.
  double margin;

  //! Locally-stored value of takeMean parameter.
  bool takeMean;
}; // class CosineEmbeddingLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "cosine_embedding_loss_impl.hpp"

#endif