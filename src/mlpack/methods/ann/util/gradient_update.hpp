/**
 * @file methods/ann/util/gradient_update.hpp
 * @author Marcus Edel
 *
 * Definition of the GradientUpdate() function which assignes a portion of the
 * given gradient to the layer/sub-layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_UTIL_GRADIENT_UPDATE_HPP
#define MLPACK_METHODS_ANN_UTIL_GRADIENT_UPDATE_HPP

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Assign a portion of the given gradient matrix to the given layer/sub-layer.
 *
 * @tparam LayerType The type of the layer that the gradient is assigned to
 *     e.g. Linear, Convolution.
 * @tparam MathTest The type of the gradient matrix e.g. arma::Mat<double>,
 *     arma::Mat<float>.
 * @param layer The layer that the gradient is assigned to.
 * @param offset The beginning of the gradient portion we assign to the layer.
 */
template<typename LayerType, typename MatType>
size_t GradientUpdate(
    const LayerType& layer, MatType& gradient, const size_t offset)
{
  size_t size = 0;

  if (layer->Parameters().n_elem > 0)
  {
    layer->Gradient() = arma::mat(gradient.memptr() + offset,
        layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);
    size += layer->Parameters().n_elem;
  }

  if (layer->Model().size() > 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
      size += GradientUpdate(layer->Model()[i], gradient, offset + size);
  }

  return size;
}

} // namespace ann
} // namespace mlpack

#endif
