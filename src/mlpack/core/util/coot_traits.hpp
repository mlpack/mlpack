/**
 * @file core/util/coot_traits.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Some traits used for template metaprogramming (SFINAE) with Bandicoot types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_COOT_TRAITS_HPP
#define MLPACK_CORE_UTIL_COOT_TRAITS_HPP

#if defined(MLPACK_HAS_COOT)

// Get whether or not the given type is any Bandicoot type
// This includes dense and cube types
template<typename T>
struct IsCoot
{
  constexpr static bool value = coot::is_coot_type<T>::value ||
                                coot::is_coot_cube_type<T>::value;
};

template<typename eT>
struct GetCubeType<coot::Mat<eT>>
{
  using type = coot::Cube<eT>;
};

template<typename eT>
struct IsVector<coot::Col<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<coot::Row<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<coot::subview_col<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsVector<coot::subview_row<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsMatrix<coot::Mat<eT> >
{
  static const bool value = true;
};

template<typename eT>
struct IsCube<coot::Cube<eT> >
{
  static const bool value = true;
};

#else

template<typename T>
struct IsCoot
{
  constexpr static bool value = false;
};

#endif // defined(MLPACK_HAS_COOT)

#endif
