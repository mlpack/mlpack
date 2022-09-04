/**
 * @file core/math/log_add.hpp
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
 * Log-sum a vector of log values.  (T should be an Armadillo type.)
 *
 * @param x vector of log values
 * @return log(e^x0 + e^x1 + ...)
 */
template<typename T>
typename T::elem_type AccuLog(const T& x);

/**
 * Compute the sum of exponentials of each element in each column, then compute
 * the log of that.  If InPlace is true, then the values of `y` will also be
 * added to the sum.
 *
 * That is, if InPlace is false, then this method will set `y` such that:
 *
 *     `y_i = log(sum(exp(x.col(i))))`
 *
 * and if InPlace is true, then `y` will be set such that:
 *
 *     `y_i = log(sum(exp(x.col(i))) + exp(y_i))`.
 */
template<typename T, bool InPlace = false>
void LogSumExp(const T& x, arma::Col<typename T::elem_type>& y);

/**
 * Compute the sum of exponentials of each element in each row, then compute the
 * log of that.  If InPlace is true, then the values of `y` will also be added
 * to the sum.
 *
 * That is, if InPlace is false, then this method will set `y` such that:
 *
 *     `y_i = log(sum(exp(x.row(i))))`
 *
 * and if InPlace is true, then `y` will be set such that:
 *
 *     `y_i = log(sum(exp(x.row(i))) + exp(y_i))`.
 */
template<typename T, bool InPlace = false>
void LogSumExpT(const T& x, arma::Col<typename T::elem_type>& y);

} // namespace mlpack

// Include implementation.
#include "log_add_impl.hpp"

#endif
