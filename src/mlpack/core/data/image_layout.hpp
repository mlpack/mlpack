/**
 * @file core/data/image_layout.hpp
 * @author Andrew Furey
 *
 * Image layout conversion utility functions.
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

 * `data::Load()` returns a matrix where each column represents an image.
 * The rows of each image represent pixel values whose channels are stored
 * consecutively, e.g [r, g, b, r, g, b, ... ]. Some mlpack functionality
 * such as convolutions require that each channel of the image be grouped
 * together instead, e.g [r, r, ... , g, g, ... , b, b].

 * 'GroupChannels()` makes a copy of the image and returns the same image
 * except the channels are grouped together instead of stored consecutively.
 *
 * @param image Image matrix whose channels are consecutive.
 * @param info  ImageInfo describing shape of image.
 * @return Mat  Copy of image whose channels are grouped together.
 */
template <typename eT>
inline arma::Mat<eT> GroupChannels(const arma::Mat<eT>& image,
    const mlpack::data::ImageInfo& info);

/**
 * The inverse of `GroupChannels()`. Given an image where each channel is
 * grouped together, make a copy and store the channels consecutively.
 *
 * @param image Image matrix in mlpack layout.
 * @param info  ImageInfo describing shape of image.
 * @return Mat  Copy of image whose channels stored consecutively.
 */
template <typename eT>
inline arma::Mat<eT> InterleaveChannels(const arma::Mat<eT>& image,
    const mlpack::data::ImageInfo& info);

} // namespace data
} // namespace mlpack

#include "image_layout_impl.hpp"

#endif
