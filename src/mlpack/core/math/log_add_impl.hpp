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

/** Internal log-addition
 *
 * @f[
 * e^z = e^x + e^y
 * e^z = e^x(1 + e^{y-x})      = e^y(1 + e^{x-y})
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

  if (std::isinf(d) || std::isinf(r))
    return r;

  return r + std::log(1 + std::exp(d));
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
  typename T::elem_type maxVal = max(x);
  if (maxVal == -std::numeric_limits<typename T::elem_type>::infinity())
    return maxVal;

  return maxVal + std::log(sum(exp(x - maxVal)));
}

/**
 * Compute the sum of exponentials of each element in each column, then compute
 * the log of that.  If InPlace is true, then the values of `y` will also be
 * added to the sum.
 */
template<typename T, bool InPlace>
void LogSumExp(const T& x, arma::Col<typename T::elem_type>& y)
{
  arma::Col<typename T::elem_type> maxs;

  if (InPlace)
  {
    // Compute the maximum in each column (treating y as a column too).
    maxs = max(max(x, 1), y);

    y = maxs + log(sum(exp(x - repmat(maxs, 1, x.n_cols)), 1) + exp(y - maxs));
  }
  else
  {
    // Compute the maximum element in each column.
    maxs = max(x, 1);

    y = maxs + log(sum(exp(x - repmat(maxs, 1, x.n_cols)), 1));
  }

  if (maxs.has_inf())
  {
    y.replace(-std::numeric_limits<typename T::elem_type>::quiet_NaN(),
              -std::numeric_limits<typename T::elem_type>::infinity());
  }
}

/**
 * Compute the sum of exponentials of each element in each row, then compute the
 * log of that.  If InPlace is true, then the values of `y` will also be added
 * to the sum.
 */
template<typename T, bool InPlace>
void LogSumExpT(const T& x, arma::Col<typename T::elem_type>& y)
{
  arma::Row<typename T::elem_type> maxs;

  if (InPlace)
  {
    // Compute the maximum element in each column.
    maxs = max(max(x, 0), y.t());

    y = maxs.t() + log(sum(exp(x - repmat(maxs, x.n_rows, 1)), 0) +
        exp(y.t() - maxs)).t();
  }
  else
  {
    // Compute the maximum element in each column.
    arma::Row<typename T::elem_type> maxs = max(x, 0);

    y = (maxs + log(sum(exp(x - repmat(maxs, x.n_rows, 1)), 0))).t();
  }

  if (maxs.has_inf())
  {
    y.replace(-std::numeric_limits<typename T::elem_type>::quiet_NaN(),
              -std::numeric_limits<typename T::elem_type>::infinity());
  }
}

} // namespace mlpack

#endif
