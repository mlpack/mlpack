/**
 * @file methods/ann/util/deterministic_update.hpp
 * @author Marcus Edel
 *
 * Definition of the DeterministicUpdate() function to update the layer and
 * sub-layer training/testing state.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_DETERMINISTIC_UPDATE_HPP
#define MLPACK_METHODS_ANN_UTIL_DETERMINISTIC_UPDATE_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Update the deterministic parameter for the given layer and all sub-layer
 * with the specified value.
 *
 * @note During training you should set the deterministic parameter for each
 * layer to false and during testing you should set deterministic to true.
 *
 * @tparam LayerType The type of the given layer e.g. Dropout, DropConnect.
 * @param layer The layer (including sub-layer) to be updated.
 * @param deterministic The training/testing state,
 *     training = false, testing = true.
 */
template<typename LayerType>
void DeterministicUpdate(const LayerType& layer, const bool deterministic)
{
  layer->Deterministic() = deterministic;

  if (layer->Model().size() > 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
      DeterministicUpdate(layer->Model()[i], deterministic);
  }
}

} // namespace ann
} // namespace mlpack

#endif
