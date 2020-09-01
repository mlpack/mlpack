/**
 * @file core/math/log_add_impl.hpp
 * @author Arash Abghari
 *
 * Implementation of logarithmic addition functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_LOG_ADD_IMPL_HPP
#define MLPACK_CORE_MATH_LOG_ADD_IMPL_HPP

#include "log_add.hpp"

namespace mlpack {
namespace math {

/** Internal log-addition
 *
 * @f[
 * e^z = e^x + e^y
 * e^z = e^x(1 + e^{y-x})     = e^y(1 + e^{x-y})
 * z   = x + \log(1 + e^{y-x}) = y + \log(1 + e^{x-y})
 * @f]
 *
 * So when \f$ y > x \f$, \f$ z = y + \log(1 + e^{x-y}) \f$;
 *    when \f$ x > y \f$, \f$ z = x + \log(1 + e^{y-x}) \f$.
 *
 * @param x log value
 * @param y log value
 * @return log(e^x + e^y)
 */
template<typename T>
T LogAdd(T x, T y)
{
  T d, r;
  if (x > y)
  {
    d = y - x;
    r = x;
  }
  else
  {
    d = x - y;
    r = y;
  }

  return (r == -std::numeric_limits<T>::infinity() ||
          d == -std::numeric_limits<T>::infinity()) ? r : r + log(1 + exp(d));
}

/**
 * Sum a vector of log values.
 *
 * @param x vector of log values
 * @return log(e^x0 + e^x1 + ...)
 */
template<typename T>
typename T::elem_type AccuLog(const T& x)
{
  typename T::elem_type sum =
      -std::numeric_limits<typename T::elem_type>::infinity();

  for (size_t i = 0; i < x.n_elem; ++i)
  {
    sum = LogAdd(sum, x[i]);
  }

  return sum;
}

} // namespace math
} // namespace mlpack

#endif
