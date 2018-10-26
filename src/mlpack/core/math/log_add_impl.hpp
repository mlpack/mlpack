/**
 * @file log_add_impl.hpp
 *
 * Implementation of logarithmic addition functions.
 */
#ifndef MLPACK_CORE_MATH_LOG_ADD_IMPL_HPP
#define MLPACK_CORE_MATH_LOG_ADD_IMPL_HPP

#include "log_add.hpp"

namespace mlpack {
namespace math {

 /** Internal log-addition
 *
 * e^z = e^x + e^y
 * e^z = e^x(1 + e^{y-x})     = e^y(1 + e^{x-y})
 * z   = x + log(1 + e^{y-x}) = y + log(1 + e^{x-y})
 * 
 * So when y > x, z = y + log(1 + e^{x-y})
 *    when x > y, z = x + log(1 + e^{y-x})
 * 
 * @param x log value
 * @param y log value
 * @return log(e^x + e^y)
 */
template<typename T>
T LogAdd(T x, T y)
{
  double d, r;
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

  for (auto&& v : x)
  {
    sum = LogAdd(sum, v);
  }

  return sum;
}

} // namespace math
} // namespace mlpack

#endif
