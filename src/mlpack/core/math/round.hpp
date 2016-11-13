/**
 * @file round.hpp
 * @author Ryan Curtin
 *
 * Implementation of round() for use on Visual Studio, where C99 isn't
 * implemented.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_ROUND_HPP
#define MLPACK_CORE_MATH_ROUND_HPP

// _MSC_VER should only be defined for Visual Studio, which doesn't implement
// C99.
#ifdef _MSC_VER

// This function ends up going into the global namespace, so it can be used in
// place of C99's round().

//! Round a number to the nearest integer.
inline double round(double a)
{
  return floor(a + 0.5);
}

#endif

#endif
