/**
 * @file methods/ann/util/reset_update.hpp
 * @author Marcus Edel
 *
 * Definition of the ResetUpdate() function which resets the layer state.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_RESET_UPDATE_HPP
#define MLPACK_METHODS_ANN_UTIL_RESET_UPDATE_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Call the Reset() function for the given layer/sub-layer.
 *
 * @tparam LayerType The type of the layer that the Reset() function is called.
 * @param layer The layer for which the Reset() function is called.
 */
template<typename LayerType>
void ResetUpdate(const LayerType& layer)
{
  layer->Reset();

  if (layer->Model().size() > 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
      ResetUpdate(layer->Model()[i]);
  }
}

} // namespace ann
} // namespace mlpack

#endif
