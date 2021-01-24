/**
 * @file methods/ann/util/output_height_update.hpp
 * @author Marcus Edel
 *
 * Definition of the OutputHeight() function that returns the output height of
 * the given layer/sub-layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_OUTPUT_HEIGHT_UPDATE_HPP
#define MLPACK_METHODS_ANN_UTIL_OUTPUT_HEIGHT_UPDATE_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Return the output height of the given layer/sub-layer.
 *
 * @tparam LayerType The layer type to get the output height for
 *     e.g. MeanPooling.
 * @param layer The layer to get the output height for.
 * @return The layer output height.
 */
template<typename LayerType>
size_t OutputHeight(const LayerType& layer)
{
  if (layer->Model().size() > 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      const size_t width = OutputHeight(layer->Model()[
          layer->Model().size() - i - 1]);

      if (width != 0)
        return width;
    }
  }

  return layer->OutputHeight();
}

} // namespace ann
} // namespace mlpack

#endif
