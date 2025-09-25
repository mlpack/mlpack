/**
 * @file core/data/image_layout.hpp
 * @author Andrew Furey
 *
 * Convert image layouts between mlpack and stb.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_LAYOUT_HPP
#define MLPACK_CORE_DATA_IMAGE_LAYOUT_HPP

#include "image_info.hpp"

namespace mlpack {
namespace data {

/**
 * Changes the `image` layout from stb layout to mlpack layout.
 *
 * @param image Image matrix in STB layout.
 * @param info  ImageInfo describing shape of image.
 */
template <typename eT>
inline arma::Mat<eT> ImageLayout(const arma::Mat<eT>& image,
    const mlpack::data::ImageInfo& info);

/**
 * Changes the `image` layout from mlpack layout to stb layout.
 *
 * @param image Image matrix in mlpack layout.
 * @param info  ImageInfo describing shape of image.
 */
template <typename eT>
inline arma::Mat<eT> STBLayout(const arma::Mat<eT>& image,
    const mlpack::data::ImageInfo& info);

} // namespace data
} // namespace mlpack

#include "image_layout_impl.hpp"

#endif
