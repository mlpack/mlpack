/**
 * @file surrogate_loss.hpp
 * @author Xiaohong Ji
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_SURROGATE_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_SURROGATE_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "surrogate_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SurrogateLoss<InputDataType, OutputDataType>::SurrogateLoss()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SurrogateLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
