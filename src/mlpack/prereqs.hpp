/**
 * @file prereqs.hpp
 *
 * The core includes that mlpack expects; standard C++ includes and Armadillo.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_PREREQS_HPP
#define __MLPACK_PREREQS_HPP

// First, check if Armadillo was included before, warning if so.
#ifdef ARMA_INCLUDES
#pragma message "Armadillo was included before mlpack; this can sometimes cause\
problems.  It should only be necessary to include <mlpack/core.hpp> and not\
<armadillo>."
#endif

// Next, standard includes.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <iostream>

// Defining _USE_MATH_DEFINES should set M_PI.
#define _USE_MATH_DEFINES
#include <math.h>

// For tgamma().
#include <boost/math/special_functions/gamma.hpp>

// But if it's not defined, we'll do it.
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
#endif

// Give ourselves a nice way to force functions to be inline if we need.
#define force_inline
#if defined(__GNUG__) && !defined(DEBUG)
  #undef force_inline
  #define force_inline __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(DEBUG)
  #undef force_inline
  #define force_inline __forceinline
#endif

// Now include Armadillo through the special mlpack extensions.
#include <mlpack/core/arma_extend/arma_extend.hpp>

#endif
