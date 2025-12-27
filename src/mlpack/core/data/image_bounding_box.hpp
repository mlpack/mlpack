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
namespace data {

template <typename ImageType, typename BoundingBoxesType>
inline void BoundingBoxImage(ImageType& src,
  const ImageInfo& srcOpt,
  const BoundingBoxesType& bbox,
  const std::string& className,
  const typename ImageType::elem_type red,
  const typename ImageType::elem_type green,
  const typename ImageType::elem_type blue,
  const size_t borderSize,
  const size_t letterSize);

} // namespace data
} // namespace mlpack

#include "image_bounding_box_impl.hpp"

#endif
