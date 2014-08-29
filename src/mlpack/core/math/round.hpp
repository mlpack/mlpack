/**
 * @file round.hpp
 * @author Ryan Curtin
 *
 * Implementation of round() for use on Visual Studio, where C99 isn't
 * implemented.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_MATH_ROUND_HPP
#define __MLPACK_CORE_MATH_ROUND_HPP

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
