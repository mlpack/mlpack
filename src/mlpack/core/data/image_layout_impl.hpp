/**
 * @file core/data/image_layout_impl.hpp
 * @author Andrew Furey
 *
 * Convert image layouts between mlpack and stb.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_IMAGE_LAYOUT_IMPL_HPP
#define MLPACK_CORE_DATA_IMAGE_LAYOUT_IMPL_HPP

#include "image_layout.hpp"
#include "image_info.hpp"

namespace mlpack {
namespace data {

template <typename eT>
arma::Mat<eT> ImageLayout(const arma::Mat<eT>& image,
    const mlpack::data::ImageInfo& info)
{
  arma::Mat<eT> output =
    arma::vectorise(
      arma::reshape(image, info.Channels(), info.Height() * info.Width()).t()
    );
  return output;
}

template <typename eT>
arma::Mat<eT> STBLayout(const arma::Mat<eT>& image,
    const mlpack::data::ImageInfo& info)
{
  arma::Mat<eT> output =
    arma::vectorise(
      arma::reshape(image, info.Height() * info.Width(), info.Channels()).t()
    );
  return output;
}

} // namespace data
} // namespace mlpack

#endif
