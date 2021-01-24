/**
 * @file methods/ann/util/loss_update.hpp
 * @author Marcus Edel
 *
 * Definition of the LossUpdate() function which returns the layer/sub-layer
 * loss.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_LOSS_UPDATE_HPP
#define MLPACK_METHODS_ANN_UTIL_LOSS_UPDATE_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Get the los from the given layer/sub-layer.
 *
 * @tparam Layer The type of the given layer.
 * @param layer The layer to get the loss for.
 * @return The layer loss.
 */
template<typename LayerType>
double LossUpdate(const LayerType& layer)
{
  double loss = layer->Loss();

  if (layer->Model().size() > 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
      loss += LossUpdate(layer->Model()[i]);
  }

  return loss;
}

} // namespace ann
} // namespace mlpack

#endif
