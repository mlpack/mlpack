/**
 * @file core/util/rand.hpp
 * @author Marcus Edel
 * @author Omar Shrit
 *
 * A simple `randX` wrapper that based on the data type and the function type,
 * e.g., randi, randn, randu forwards to coot::randX` or `arma::randX`. This
 * file is initially copied from ensmallen
 *
 * mlpack is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_RAND_HPP
#define MLPACK_CORE_UTIL_RAND_HPP

namespace mlpack {

#ifdef MLPACK_HAS_COOT

template<typename InputType, typename DistrParam>
InputType Randi(const size_t rows,
                const size_t cols,
                const DistrParam& param
                const typename std::enable_if_t<
                    coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::randi<InputType>(rows, cols, param);
}

template<typename InputType, typename DistrParam>
InputType Randi(const size_t n_elem,
                const DistrParam& param
                const typename std::enable_if_t<
                    coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::randi<InputType>(n_elem, param);
}

template<typename InputType>
InputType Randn(const size_t rows, 
                const size_t cols, 
                const typename std::enable_if_t<
                    coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::randn<InputType>(rows, cols);
}

template<typename InputType, typename DistrParam>
InputType Randn(const size_t rows,
                const size_t cols,
                const DistrParam& param
                const typename std::enable_if_t<
                    coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::randn<InputType>(rows, cols, param);
}

template<typename InputType>
InputType Randn(const size_t n_elem, 
                const typename std::enable_if_t<
                    coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::randn<InputType>(n_elem);
}

template<typename InputType>
InputType Randu(const size_t rows,
                const size_t cols,
                const typename std::enable_if<
                    coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::randu<InputType>(rows, cols);
}

template<typename InputType>
InputType Randu(const size_t n_elem, 
                const typename std::enable_if_t<
                    coot::is_coot_type<InputType>::value>* = 0)
{
  return coot::randu<InputType>(n_elem);
}

#endif

template<typename InputType, typename DistrParam>
InputType Randi(const size_t rows,
                const size_t cols,
                const DistrParam& param,
                const typename std::enable_if_t<
                    arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::randi<InputType>(rows, cols, param);
}

template<typename InputType, typename DistrParam>
InputType Randi(const size_t n_elem,
                const DistrParam& param,
                const typename std::enable_if_t<
                    arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::randi<InputType>(n_elem, param);
}

template<typename InputType, typename VarType, typename VarType2>
InputType Randn(const VarType rows,
                const VarType2 cols,
                const typename std::enable_if_t<
                    arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::randn<InputType>(rows, cols);
}

template<typename InputType, typename VarType, typename VarType2,
    typename DistrParam>
InputType Randn(const VarType rows,
                const VarType2 cols,
                const DistrParam& param,
                const typename std::enable_if_t<
                    arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::randn<InputType>(rows, cols, param);
}

template<typename InputType>
InputType Randn(const size_t n_elem,
                const typename std::enable_if_t<
                    arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::randn<InputType>(n_elem);
}

template<typename InputType>
InputType Randu(const size_t rows,
                const size_t cols,
                const typename std::enable_if_t<
                    arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::randu<InputType>(rows, cols);
}

template<typename InputType>
InputType Randu(const size_t n_elem,
                const typename std::enable_if_t<
                    arma::is_arma_type<InputType>::value>* = 0)
{
  return arma::randu<InputType>(n_elem);
}

} // namespace mlpack

#endif
