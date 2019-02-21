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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
TripletLoss<InputDataType, OutputDataType>
::TripletLoss(const double margin) : margin(margin)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
