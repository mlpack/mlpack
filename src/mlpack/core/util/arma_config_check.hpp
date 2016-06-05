/**
 * @file arma_config_check.hpp
 * @author Ryan Curtin
 *
 * Using the contents of arma_config.hpp, try to catch the condition where the
 * user has included mlpack with ARMA_64BIT_WORD enabled but mlpack was compiled
 * without ARMA_64BIT_WORD enabled.  This should help prevent a long, drawn-out
 * debugging process where nobody can figure out why the stack is getting
 * mangled.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_CORE_UTIL_ARMA_CONFIG_CHECK_HPP
#define MLPACK_CORE_UTIL_ARMA_CONFIG_CHECK_HPP

#include "arma_config.hpp"

#ifdef ARMA_64BIT_WORD
  #ifdef MLPACK_ARMA_NO_64BIT_WORD
    #pragma message "mlpack was compiled without ARMA_64BIT_WORD, but you are \
compiling with ARMA_64BIT_WORD.  This will almost certainly cause irreparable \
disaster.  Either disable ARMA_64BIT_WORD in your application which is using \
mlpack, or, recompile mlpack against a version of Armadillo which has \
ARMA_64BIT_WORD enabled."
  #endif
#else
  #ifdef MLPACK_ARMA_64BIT_WORD
    #pragma message "mlpack was compiled with ARMA_64BIT_WORD, but you are \
compiling without ARMA_64BIT_WORD.  This will almost certainly cause \
irreparable disaster.  Either enable ARMA_64BIT_WORD in your application which \
is using mlpack, or, recompile mlpack against a version of Armadillo which has \
ARMA_64BIT_WORD disabled."
  #endif
#endif

#endif
