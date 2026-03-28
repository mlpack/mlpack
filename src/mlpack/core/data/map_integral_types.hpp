/**
 * @file core/data/map_integral_types.hpp
 * @author Omar Shrit
 * @author Ryan Curtin
 *
 * Convert between different integer types including different signes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MAP_INTEGRAL_TYPES_HPP
#define MLPACK_CORE_DATA_MAP_INTEGRAL_TYPES_HPP

#include <type_traits>

namespace mlpack {

/*
 * Given two integral types eT2 and eT1, where eT2 is always signed,
 * convert to eT1 such that the range of the values in `target` is
 * [0, numeric_limits<eT>::max()] for unsigned eT1, and
 * [numeric_limits<eT>::min(), numeric_limits<eT>::max()] for signed eT1.
 */
template<typename eT1, typename eT2>
inline void MapSignedIntegralTypes(arma::Mat<eT1>& target, arma::Mat<eT2>& src)
{
  // If the target type is smaller than the loaded type, then we need to shrink
  // the range before the conversion.
  if constexpr (sizeof(eT1) < sizeof(eT2))
  {
    src /= std::pow(2, 8 * (sizeof(eT2) - sizeof(eT1)));
  }

  // If the target type is unsigned, then we have to reinterpret the samples and
  // apply a shift, because Armadillo's `conv_to` will truncate negative values
  // to 0.
  if constexpr (!std::is_signed_v<eT1>)
  {
    typedef std::make_unsigned_t<eT2> ueT2;
    arma::Mat<ueT2> reinterpretedSrc((ueT2*) src.memptr(), src.n_rows,
        src.n_cols, false);
    reinterpretedSrc += std::pow(2, 8 * sizeof(ueT2) - 1);

    target = arma::conv_to<arma::Mat<eT1>>::from(std::move(reinterpretedSrc));
  }
  else
  {
    target = arma::conv_to<arma::Mat<eT1>>::from(src);
  }

  // If the target type is larger than the loaded type, then we need to expand
  // the range after the conversion.
  if constexpr (sizeof(eT1) > sizeof(eT2))
  {
    target *= std::pow(2, 8 * (sizeof(eT1) - sizeof(eT2)));
  }
}

/*
 * Given two integral types eT2 and eT1, where eT2 can be signed or unsigned,
 * convert to eT1 such that the range of the values in `target` is
 * [numeric_limits<eT>::min(), numeric_limits<eT>::max()].
 */
template<typename eT1, typename eT2>
inline void MapUnsignedIntegralTypes(arma::Mat<eT1>& target,
                                     arma::Mat<eT2>& src)
{
  // If the source is unsigned, shift it to the signed range first.
  if constexpr (!std::is_signed_v<eT2>)
  {
    typedef std::make_signed_t<eT2> seT2;
    arma::Mat<seT2> signedSrc((seT2*) src.memptr(), src.n_rows,
        src.n_cols, false);
    signedSrc -= static_cast<seT2>(std::pow(2, 8 * sizeof(eT2) - 1));

    if constexpr (sizeof(eT1) < sizeof(seT2))
    {
      signedSrc /= static_cast<seT2>(
          std::pow(2, 8 * (sizeof(seT2) - sizeof(eT1))));
    }

    target = arma::conv_to<arma::Mat<eT1>>::from(signedSrc);

    if constexpr (sizeof(eT1) > sizeof(seT2))
    {
      target *= static_cast<eT1>(
          std::pow(2, 8 * (sizeof(eT1) - sizeof(seT2))));
    }
  }
  else
  {
    if constexpr (sizeof(eT1) < sizeof(eT2))
    {
      src /= static_cast<eT2>(
          std::pow(2, 8 * (sizeof(eT2) - sizeof(eT1))));
    }

    target = arma::conv_to<arma::Mat<eT1>>::from(src);

    if constexpr (sizeof(eT1) > sizeof(eT2))
    {
      target *= static_cast<eT1>(
          std::pow(2, 8 * (sizeof(eT1) - sizeof(eT2))));
    }
  }
}

} //namespace mlpack

#endif
