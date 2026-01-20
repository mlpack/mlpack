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

template <typename MatType>
inline void UpdatePixel(MatType& src,
  const ImageInfo& srcOpt,
  const size_t x,
  const size_t y,
  const MatType& color)
{
  const size_t redChannel =
    x * srcOpt.Channels() + y * srcOpt.Channels() * srcOpt.Width();
  src.rows(redChannel, redChannel + color.n_rows - 1) =  color;
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
      const bool on =
        !(font8x8Basic[(size_t)letter][i] & (unsigned char)(1 << j));
      const MatType set = arma::repmat(MatType({ on }), srcOpt.Channels(), 1);
      for (size_t k = 0; k < size * size; k++)
      {
        const size_t px = x + (j * size) + (k % size);
        const size_t py = y + (i * size) + (k / size);
        UpdatePixel(src, srcOpt, px, py, set);
      }
    }
  }
}

template <typename ImageType, typename BoundingBoxesType>
inline void BoundingBoxImage(ImageType& src,
  const ImageInfo& srcOpt,
  const BoundingBoxesType& bbox,
  const ImageType& color,
  const size_t borderSize,
  const std::string& className,
  const size_t letterSize)
{
  using ElemType = typename BoundingBoxesType::elem_type;

  const size_t imageSize = srcOpt.Width() * srcOpt.Height() * srcOpt.Channels();
  if (src.n_elem != imageSize) {
    std::ostringstream errMessage;
    errMessage << "BoundingBoxImage(): The size of the image (" << src.n_elem
               << ") does not match the given dimensions ("
               << srcOpt.Width() << ", " << srcOpt.Height() << ", "
               << srcOpt.Channels() << ").";
    throw std::logic_error(errMessage.str());
  }

  if (color.n_rows != srcOpt.Channels() || color.n_cols != 1) {
    std::ostringstream errMessage;
    errMessage << "BoundingBoxImage(): The color vector of shape ("
               << color.n_rows << ", " << color.n_cols << ") does not match "
               << "expected shape (" << srcOpt.Channels() << ", 1)";
    throw std::logic_error(errMessage.str());
  }

  if (bbox.n_rows < 4) {
    std::ostringstream errMessage;
    errMessage << "BoundingBoxImage(): A bounding box is made up of 4 points "
               "but was given " << color.n_rows;
    throw std::logic_error(errMessage.str());
  }

  const ElemType maxWidth = srcOpt.Width() - 1;
  const ElemType maxHeight = srcOpt.Height() - 1;
  const ElemType x1 = std::clamp<ElemType>(bbox(0), 0, maxWidth);
  const ElemType y1 = std::clamp<ElemType>(bbox(1), 0, maxHeight);
  const ElemType x2 = std::clamp<ElemType>(bbox(2), 0, maxWidth);
  const ElemType y2 = std::clamp<ElemType>(bbox(3), 0, maxHeight);

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
      const size_t yTop = y1 + b;
      const size_t yBottom = y2 - b;
      UpdatePixel(src, srcOpt, x, yTop, color);
      UpdatePixel(src, srcOpt, x, yBottom, color);
    }
    for (int y = y1; y <= y2; y++)
    {
      const size_t xLeft = x1 + b;
      const size_t xRight = x2 - b;
      UpdatePixel(src, srcOpt, xLeft, y, color);
      UpdatePixel(src, srcOpt, xRight, y, color);
    }
  }

  // Draw class name
  if (letterSize == 0)
    return;
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

} // namespace mlpack

#endif
