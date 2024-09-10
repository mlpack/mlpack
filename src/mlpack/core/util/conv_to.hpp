/**
 * @file core/util/conv_to.hpp
 * @author Marcus Edel
 * @author Omar Shrit
 *
 * A simple `conv_to` wrapper that based on the data type forwards to
 * `coot::conv_to` or `arma::conv_to`. This file is borrowed from ensmallen.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_CONV_TO_HPP
#define MLPACK_CORE_UTIL_CONV_TO_HPP

namespace mlpack {

/**
 * A utility class that based on the data type forwards to `coot::conv_to` or
 * `arma::conv_to`.
 *
 * @tparam OutputType The data type to convert to.
 */
template<typename OutputType>
class ConvTo
{
 public:
#ifdef MLPACK_HAS_COOT
  /**
   * Convert from one matrix type to another by forwarding to `coot::conv_to`.
   *
   * @param input The input that is converted.
   */
  template<typename InputType>
  inline static OutputType From(const InputType& input,
                                const typename std::enable_if_t<
                                    coot::is_coot_type<InputType>::value ||
                                    coot::is_coot_type<OutputType>::value>* = 0)
  {
    return coot::conv_to<OutputType>::from(input);
  }
#endif

  /**
   * Convert from one matrix type to another by forwarding to `arma::conv_to`.
   *
   * @param input The input that is converted.
   */
  template<typename InputType>
  inline static OutputType From(const InputType& input,
                                const typename std::enable_if_t<
                                    arma::is_arma_type<InputType>::value ||
                                    arma::is_arma_type<OutputType>::value>* = 0)
  {
    return arma::conv_to<OutputType>::from(input);
  }
};

} // namespace mlpack

#endif
