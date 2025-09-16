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
  arma::Mat<eT> input =
    arma::reshape(image, info.Channels(), info.Height() * info.Width()).t();
  arma::Mat<eT> output(info.Width() * info.Height(), info.Channels(),
    arma::fill::none);

  for (size_t i = 0; i < input.n_cols; i++)
  {
    arma::Mat<eT> col(input.col(i));
    col.reshape(info.Width(), info.Height());
    col = col.t();
    output.col(i) = arma::vectorise(col);
  }
  return arma::vectorise(output);
}

template <typename eT>
arma::Mat<eT> STBLayout(const arma::Mat<eT>& image,
    const mlpack::data::ImageInfo& info)
{
  arma::Mat<eT> input(image);
  input.reshape(info.Width() * info.Height(), info.Channels());
  for (size_t i = 0; i < input.n_cols; i++) {
    arma::Mat<eT> col(input.col(i));
    col.reshape(info.Height(), info.Width());
    col = col.t();
    input.col(i) = arma::vectorise(col);
  }

  return arma::reshape(input.t(), info.Channels() * info.Height() *
                       info.Width(), 1);
}

} // namespace data
} // namespace mlpack

#endif
