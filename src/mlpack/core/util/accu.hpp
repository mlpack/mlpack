/**
 * @file core/util/accu.hpp
 * @author Omar Shrit
 *
 * A simple `Accu` wrapper that based on the data type forwards to
 * `coot::accu` or `arma::accu`.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_ACCU_HPP
#define MLPACK_CORE_UTIL_ACCU_HPP

namespace mlpack {


#ifdef MLPACK_HAS_COOT

  /**
   * Compute the sum of all the elements in InputType
   *
   * @param input The input type to be a bandicoot vector, matrix or cube.
   */
  template<typename InputType>
  typename InputType::elem_type Accu(const InputType& input, 
                                     const typename std::enable_if_t<
                                     coot::is_coot_type<InputType>::value>* = 0)
  {
    return coot::accu(input);
  }

#endif

  /**
   * Compute the sum of all the elements in InputType
   *
   * @param input The input type to be an armadillo vector, matrix or cube.
   */
  template<typename InputType>
  typename InputType::elem_type Accu(const InputType& input, 
                                     const typename std::enable_if_t<
                                     arma::is_arma_type<InputType>::value>* = 0)
  {
    return arma::accu(input);
  }

} // namespace mlpack

#endif
