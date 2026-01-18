/**
 * @file core/data/image_bounding_box_impl.hpp
 * @author Andrew Furey
 *
 * Draw bounding boxes and labels onto images.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_BOUNDING_BOX_IMPL_HPP
#define MLPACK_CORE_DATA_IMAGE_BOUNDING_BOX_IMPL_HPP

#include "image_bounding_box.hpp"
#include "image_options.hpp"

namespace mlpack {
namespace data {

template <typename MatType>
inline void UpdatePixel(MatType& src,
  const ImageInfo& srcOpt,
  const size_t x,
  const size_t y,
  const MatType& color)
{
  const size_t redChannel =
    x * srcOpt.Channels() + y * srcOpt.Channels() * srcOpt.Width();

  src.at(redChannel, 0) = color.at(0); // TODO: use submat, color needs to be a column vector
  src.at(redChannel + 1, 0) = color.at(1);
  src.at(redChannel + 2, 0) = color.at(2);
}

template <typename MatType>
inline void DrawLetter(MatType& src,
  const ImageInfo& srcOpt,
  const char letter,
  const typename MatType::elem_type x,
  const typename MatType::elem_type y,
  const size_t size)
{
  const size_t fontWidth = 8;
  for (size_t i = 0; i < fontWidth; i++)
  {
    for (size_t j = 0; j < fontWidth; j++)
    {
      const bool set = !(font8x8_basic[letter][i] & (unsigned char)(1 << j));
      for (size_t k = 0; k < size * size; k++)
      {
        const size_t px = x + (j * size) + (k % size);
        const size_t py = y + (i * size) + (k / size);
        UpdatePixel(src, srcOpt, px, py, set, set, set);
      }
    }
  }
}

template <typename ImageType, typename BoundingBoxesType>
inline void BoundingBoxImage(ImageType& src,
  const ImageInfo& srcOpt,
  const BoundingBoxesType& bbox,
  const std::string& className,
  const ImageType& color,
  const size_t borderSize,
  const size_t letterSize)
{
  using ElemType = typename BoundingBoxesType::elem_type;

  if (color.n_elem != srcOpt.Channels()) {
    std::ostringstream errMessage;
    errMessage << "BoundingBoxImage(): The number of color channels ("
               << color.n_elem << ") does not match the number image channels ("
               << srcOpt.Channels() << ")";
    throw std::logic_error(errMessage.str());

  }

  const ElemType x1 = bbox(0).clamp(0, srcOpt.Width() - 1);
  const ElemType y1 = bbox(1).clamp(0, srcOpt.Height() - 1);
  const ElemType x2 = bbox(2).clamp(0, srcOpt.Width() - 1);
  const ElemType y2 = bbox(3).clamp(0, srcOpt.Height() - 1);

  if (x1 >= x2)
  {
    std::ostringstream errMessage;
    errMessage << "BoundingBoxImage(): x1 should be < x2, but "
               << x1 << " >= " << x2;
    throw std::logic_error(errMessage.str());
  }

  if (y1 >= y2)
  {
    std::ostringstream errMessage;
    errMessage << "BoundingBoxImage(): y1 should be < y2, but "
               << y1 << " >= " << y2;
    throw std::logic_error(errMessage.str());
  }

  for (size_t b = 0; b < borderSize; b++)
  {
    for (size_t x = x1; x <= x2; x++)
    {
      // Top
      const size_t yT = y1 + b;
      // Bottom
      const size_t yB = y2 - b;
      // x, yT
      UpdatePixel(src, srcOpt, x, yT, color);
      UpdatePixel(src, srcOpt, x, yB, color);
    }
    for (int y = y1; y <= y2; y++)
    {
      // Left
      const size_t xL = x1 + b;
      // Right
      const size_t xR = x2 - b;
      UpdatePixel(src, srcOpt, xL, y, color);
      UpdatePixel(src, srcOpt, xR, y, color);
    }
  }

  // Draw class name
  size_t dx = x1;
  const ElemType update = letterSize * 8;
  for (size_t i = 0; i < className.size(); i++)
  {
    if (dx + update > srcOpt.Width())
      break;
    DrawLetter(src, srcOpt, className[i], dx, y1, letterSize);
    dx += update;
  }
}

} // namespace data
} // namespace mlpack

#endif
