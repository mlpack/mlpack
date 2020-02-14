/**
 * @file triplet_margin_loss_impl.hpp
 * @author Prince Gupta
 *
 * Implementation of the Triplet Margin Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_IMPL_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_IMPL_LOSS_HPP

// In case it hasn't been included.
#include "triplet_margin_loss.hpp"

namespace mlpack {
namespace ann /** Artifical Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
TripletMarginLoss<InputDataType, OutputDataType>::TripletMarginLoss(
    const double margin) : margin(margin)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename AnchorType, typename PositiveType, typename NegativeType>
double TripletMarginLoss<InputDataType, OutputDataType>::Forward(
    const AnchorType&& anchor,
    const PositiveType&& positive,
    const NegativeType&& negative)
{
  return std::max(0.0, arma::accu(arma::pow(anchor - positive, 2)) -
      arma::accu(arma::pow(anchor - negative, 2)) + margin) / anchor.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template <
    typename AnchorType,
    typename PositiveType,
    typename NegativeType,
    typename OutputType
>
void TripletMarginLoss<InputDataType, OutputDataType>::Backward(
    const AnchorType&& anchor,
    const PositiveType&& positive,
    const NegativeType&& negative,
    OutputType&& output
    )
{
  output = 2 * (negative - positive) / anchor.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void TripletMarginLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(margin);
}

} // namespace ann
} // namespace mlpack

#endif
