/**
 * @file core/data/image_bounding_box.hpp
 * @author Andrew Furey
 *
 * Draw bounding boxes and labels onto images.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_BOUNDING_BOX_HPP
#define MLPACK_CORE_DATA_IMAGE_BOUNDING_BOX_HPP

#include "image_options.hpp"
#include "font8x8_basic.h"

namespace mlpack {

template <typename ImageType,
          typename BoundingBoxesType,
          typename ColorType>
inline void BoundingBoxImage(ImageType& src,
  const ImageInfo& opts,
  const BoundingBoxesType& bbox,
  const ColorType& color,
  const size_t borderSize = 1,
  const std::string& className = "",
  const size_t letterSize = 1);

} // namespace mlpack

#include "image_bounding_box_impl.hpp"

#endif
