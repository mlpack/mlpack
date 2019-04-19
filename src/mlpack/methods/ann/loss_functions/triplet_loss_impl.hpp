/**
 * @file triplet_loss_impl.hpp
 * @author Shardul Shailendra Parab
 *
 * Implementation of the triplet loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_TRIPLET_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_TRIPLET_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "triplet_loss.hpp"
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
TripletLoss<InputDataType, OutputDataType>
::TripletLoss(const double margin) : margin(margin)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType>
double TripletLoss<InputDataType, OutputDataType>::Forward(
    const InputType&& anchor, const InputType&& positive,
    const InputType&& negative)
{
  arma::mat positive_distance = 
      metric::SquaredEuclideanDistance::Evaluate(anchor, positive);
  
  arma::mat negative_distance = 
       metric::SquaredEuclideanDistance::Evaluate(anchor, negative);

  double triplet_loss = 
      std::max(arma::accu(positive_distance - negative_distance) + margin, 0);

  return triplet_loss;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void TripletLoss<InputDataType, OutputDataType>::Backward(
    const InputType&& anchor,
    const InputType&& positive,
    const InputType&& negative,
    OutputType&& output)
{
  if (TripletLoss<InputDataType, OutputDataType>
      ::Forward(anchor, positive, negative) != 0)
  {
    float output_anchor =arma::accu((negative - positive)*2);
    float output_positive = arma::accu((anchor - positive)*(-2));
    float output_negative = arma::accu((anchor - negative)*2);
    output = arma::mat(output_anchor, output_positive, output_negative); 
  }
  else
  {
    output.zeros();
  }
  
}

} // namespace ann
} // namespace mlpack

#endif
