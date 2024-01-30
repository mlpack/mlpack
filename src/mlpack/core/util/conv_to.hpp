/**
 * @file core/util/conv_to.hpp
 * @author Marcus Edel
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
class conv_to
{
  public:

#ifdef MLPACK_HAS_COOT

    /**
    * Convert from one matrix type to another by forwarding to `coot::conv_to`.
    *
    * @param input The input that is converted.
    */
   template<typename InputType>
   inline static typename std::enable_if<
      !arma::is_arma_type<InputType>::value, OutputType>::type
   from(const InputType& input)
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
   inline static typename std::enable_if<
      arma::is_arma_type<InputType>::value, OutputType>::type
   from(const InputType& input)
   {
     return arma::conv_to<OutputType>::from(input);
   }
};

} // namespace mlpack

#endif
