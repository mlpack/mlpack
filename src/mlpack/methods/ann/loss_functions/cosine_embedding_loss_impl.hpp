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

template<typename InputDataType, typename OutputDataType>
template<
    typename FirstTensor,
    typename SecondTensor,
    typename ThirdTensor
>
double CosineEmbeddingLoss<InputDataType, OutputDataType>::Forward(
    const FirstTensor&& x1,
    const SecondTensor&& x2,
    const ThirdTensor&& y)
{
  const size_t cols = x1.n_cols;
  const size_t batchSize = x1.n_elem / cols;
  if (x1.n_rows != x2.n_rows || x1.n_cols != x2.n_cols ||
      x1.n_elem != x2.n_elem)
  {
    Log::Fatal << "Input Dimensions must be same." << std::endl;
  }

  if (y.n_elem < batchSize)
  {
    Log::Fatal << "Number of rows mismatch." << std::endl;
  }
  
  arma::colvec inputTemp1 = arma::vectorise(x1);
  arma::colvec inputTemp2 = arma::vectorise(x2);
  double loss = 0.0;

  for(size_t i = 0; i < inputTemp1.n_elem; i+=cols)
  {
    if (y(i / cols) == 1)
    {
      loss += 1 - CosineDistance(inputTemp1(arma::span(i, i + cols - 1)),
          inputTemp2(arma::span(i, i + cols - 1)));
    }
    else if (y(i / cols) == -1)
    {
      double currentLoss = CosineDistance(inputTemp1(arma::span(i, i + cols - 1)),
          inputTemp2(arma::span(i, i + cols - 1))) - margin;
      loss += currentLoss > 0 ? currentLoss : 0;
    }
    else
    {
      Log::Fatal << "y should only contain 1 and -1." << std::endl;
    }
  }

  if (takeMean)
  {
    loss = (double)loss / y.n_elem;
  }
  return loss;
}

template<typename InputDataType, typename OutputDataType>
template<
    typename FirstTensor,
    typename SecondTensor,
    typename ThirdTensor,
    typename OutputTensor
>
void CosineEmbeddingLoss<InputDataType, OutputDataType>::Backward(
    const FirstTensor&& x1,
    const SecondTensor&& x2,
    const ThirdTensor&& y,
    const OutputTensor&& output)
{
  const size_t cols = x1.n_cols;
  const size_t batchSize = x1.n_elem / cols;
  if (x1.n_rows != x2.n_rows || x1.n_cols != x2.n_cols ||
      x1.n_elem != x2.n_elem)
  {
    Log::Fatal << "Input Dimensions must be same." << std::endl;
  }

  if (y.n_elem < batchSize)
  {
    Log::Fatal << "Number of rows mismatch." << std::endl;
  }
  
  arma::colvec inputTemp1 = arma::vectorise(x1);
  arma::colvec inputTemp2 = arma::vectorise(x2);
  arma::colvec outputTemp(inputTemp1.n_elem, 1);

  for(size_t i = 0; i < inputTemp1.n_elem; i+=cols)
  {
    if (y(i / cols) != 1 && y(i / cols) != -1)
    {
      Log::Fatal << "y should only contain 1 and -1." << std::endl;
    }

    outputTemp(arma::span(i, i + cols -1)) = arma::sign(y(i / cols)) *
        (arma::normalise(inputTemp2(arma::span(i, i + cols - 1))) -
        arma::normalise(inputTemp1(arma::span(i, i + cols - 1))) *
        CosineDistance(inputTemp1(arma::span(i, i + cols - 1)),
        inputTemp2(arma::span(i, i + cols - 1)))) /
        std::sqrt(arma::accu(arma::pow(inputTemp1(arma::span(i, i + cols - 1)),
        2)));
  }
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