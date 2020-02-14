/**
 * @file cosine_embedding_loss_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of the Cosine Embedding loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_COSINE_EMBEDDING_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_COSINE_EMBEDDING_IMPL_HPP

// In case it hasn't yet been included.
#include "cosine_embedding_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
CosineEmbeddingLoss<InputDataType, OutputDataType>::CosineEmbeddingLoss(
    const double margin, const bool takeMean):
    margin(margin), takeMean(takeMean)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
template <typename InputType, typename TargetType>
double CosineEmbeddingLoss<InputDataType, OutputDataType>::Forward(
    const InputType &&x1, const InputType &&x2, const TargetType &&y)
{
  if (x1.n_rows != x2.n_rows || x1.n_cols != x2.n_cols ||
      x1.n_slices != x2.n_slices)
  {
    Log::Fatal << "Input Dimensions must be same." << std::endl;
  }

  if(y.n_elem != x1.n_rows * x1.n_slices)
  {
    Log::Fatal << "Number of rows mismatch." << std::endl;
  }
  
  arma::colvec inputTemp1 = arma::vectorise(x1);
  arma::colvec inputTemp2 = arma::vectorise(x2);
  double loss = 0.0;
  const size_t cols = x1.n_cols;
  const size_t batchSize = x1.n_rows * x1.n_slices;

  for(size_t i = 0; i < batchSize; i += cols)
  {
    if (y(i / cols) == 1)
    {
      loss += 1 - CosineDistance(inputTemp1(arma::span(i, i + cols - 1)),
          inputTemp2(arma::span(i, i + cols -1)));
    }
    else if (y(i / cols) == -1)
    {
      loss += std::max(CosineDistance(inputTemp1(arma::span(i, i + cols - 1)),
          inputTemp2(arma::span(i, i + cols -1))) - margin, 0);
    }
    else
    {
      Log::Fatal << "y should only contain 1 and -1." << std::endl;
    }
  }

  if(takeMean)
  {
    loss = (double)loss / y.n_elem;
  }
  return loss;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void CosineEmbeddingLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif