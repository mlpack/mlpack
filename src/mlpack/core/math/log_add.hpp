/**
 * @file log_add.hpp
 * @author Arash Abghari
 *
 * Functions for logarithmic addition.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_LOG_ADD_HPP
#define MLPACK_CORE_MATH_LOG_ADD_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace math {

/**
 * Internal log-addition.
 *
 * @param x log value
 * @param y log value
 * @return log(e^x + e^y)
 */
template<typename T>
T LogAdd(T x, T y);

/**
 * Sum a vector of log values.  (T should be an Armadillo type.)
 *
 * @param x vector of log values
 * @return log(e^x0 + e^x1 + ...)
 */
template<typename T>
typename T::elem_type AccuLog(const T& x);

} // namespace math
} // namespace mlpack

// Include implementation.
#include "log_add_impl.hpp"

#endif
