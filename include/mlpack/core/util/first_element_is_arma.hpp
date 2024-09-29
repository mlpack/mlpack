/**
 * @file core/util/first_element_is_arma.hpp
 * @author Ryan Curtin
 *
 * Utility struct to detect whether the first element in a parameter pack is an
 * Armadillo type.
 */
#ifndef MLPACK_CORE_UTIL_FIRST_ELEMENT_IS_ARMA_HPP
#define MLPACK_CORE_UTIL_FIRST_ELEMENT_IS_ARMA_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

// This utility struct returns the first type of a parameter pack.
template<typename... CallbackTypes>
struct First
{
  typedef void type;
};

// This matches whenever CallbackTypes has one or more elements.
template<typename T, typename... CallbackTypes>
struct First<T, CallbackTypes...>
{
  typedef T type;
};

// This utility template struct detects whether the first element in a
// parameter pack is an Armadillo type.  It is entirely for the deprecated
// constructor below and can be removed when that is removed during the
// release of mlpack 5.0.0.
template<typename... CallbackTypes>
struct FirstElementIsArma
{
  static constexpr bool value = arma::is_arma_type<
      typename std::remove_reference<
          typename First<CallbackTypes...>::type
      >::type>::value;
};

}

#endif
