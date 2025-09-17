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
  arma::Mat<eT> output(image.n_rows, image.n_cols);
  for (size_t i = 0; i < image.n_cols; i++)
  {
    arma::Mat<eT> colAlias;
    MakeAlias(colAlias, image, image.n_rows, 1, i * image.n_rows);
    output.col(i) =
      arma::vectorise(
        arma::reshape(colAlias, info.Channels(), info.Height() * info.Width())
        .t()
      );
  }
  return output;
}

template <typename eT>
arma::Mat<eT> STBLayout(const arma::Mat<eT>& image,
    const mlpack::data::ImageInfo& info)
{
  arma::Mat<eT> output(image.n_rows, image.n_cols);
  for (size_t i = 0; i < image.n_cols; i++)
  {
    arma::Mat<eT> colAlias;
    MakeAlias(colAlias, image, image.n_rows, 1, i * image.n_rows);
    output.col(i) =
      arma::vectorise(
        arma::reshape(colAlias, info.Height() * info.Width(), info.Channels())
        .t()
      );
  }
  return output;
}

} // namespace data
} // namespace mlpack

#endif
