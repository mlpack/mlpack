/**
 * @file core/data/is_naninf.hpp
 * @author Ryan Curtin
 *
 * This is an adapted version of Conrad Sanderson's implementation of
 * arma::diskio::convert_naninf() from Armadillo.  It is here so as to avoid
 * using Armadillo internal functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_HAS_NANINF_HPP
#define MLPACK_CORE_DATA_HAS_NANINF_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 * See if the token is a NaN or an Inf, and if so, set the value accordingly and
 * return a boolean representing whether or not it is.
 */
template<typename T>
inline bool IsNaNInf(T& val, const std::string& token)
{
  // See if the token represents a NaN or Inf.
  if ((token.length() == 3) || (token.length() == 4))
  {
    const bool neg = (token[0] == '-');
    const bool pos = (token[0] == '+');

    const size_t offset = ((neg || pos) && (token.length() == 4)) ? 1 : 0;

    const std::string token2 = token.substr(offset, 3);

    if ((token2 == "inf") || (token2 == "Inf") || (token2 == "INF"))
    {
      if (std::numeric_limits<T>::has_infinity)
      {
        val = (!neg) ? std::numeric_limits<T>::infinity() :
            -1 * std::numeric_limits<T>::infinity();
      }
      else
      {
        val = (!neg) ? std::numeric_limits<T>::max() :
            -1 * std::numeric_limits<T>::max();
      }

      return true;
    }
    else if ((token2 == "nan") || (token2 == "Nan") || (token2 == "NaN") ||
        (token2 == "NAN") )
    {
      if (std::numeric_limits<T>::has_quiet_NaN)
        val = std::numeric_limits<T>::quiet_NaN();
      else
        val = T(0);

      return true;
    }
  }

  return false;
}

} // namespace data
} // namespace mlpack

#endif
