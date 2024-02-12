/**
 * @file core/util/min_max.hpp
 * @author Omar Shrit
 * @author Ryan Curtin
 *
 * A shim around arma and coot functions to avoid confusion with standard 
 * library functions. This is necessary as we are using ADL to allow the
 * compiler to deduce which library the functions belong without the need
 * for namespace. This is mostly needed for MSVC compiler, gcc seems to
 * pass without an issue.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_MIN_MAX_HPP
#define MLPACK_CORE_UTIL_MIN_MAX_HPP

namespace mlpack {

#ifdef MLPACK_HAS_COOT

/**
 * Forward to coot::max if the Input type is bandicoot.
 *
 * @param input The input that is passed.
 */
template<typename InputType>
inline InputType Max(const InputType& A, const InputType& B,
                     const typename std::enable_if_t<
                     coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::max(A, B);
}

/**
 * Forward to coot::min if the Input type is bandicoot.
 *
 * @param input The input that is passed.
 */
template<typename InputType>
inline InputType Min(const InputType& A, const InputType& B, 
                     const typename std::enable_if_t<
                     coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::min(A, B);
}

#endif

/**
 * forward to arma::max if the input type is armadillo.
 *
 * @param input the input that is passed.
 */
template<typename InputType>
inline InputType Max(const InputType& A, const InputType& B, 
                     const typename std::enable_if_t<
                     arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::max(A, B);
}

/**
 * forward to arma::min if the input type is armadillo.
 *
 * @param input the input that is passed.
 */
template<typename InputType>
inline InputType Min(const InputType& A, const InputType& B, 
                     const typename std::enable_if_t<
                     arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::min(A, B);
}

} // namespace mlpack

#endif
