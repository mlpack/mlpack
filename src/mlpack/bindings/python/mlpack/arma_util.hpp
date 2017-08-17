/**
 * @file arma_util.hpp
 * @author Ryan Curtin
 *
 * Utility function for Cython to set the memory state of an Armadillo object.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_CYTHON_ARMA_UTIL_HPP
#define MLPACK_BINDINGS_PYTHON_CYTHON_ARMA_UTIL_HPP

// Include Armadillo via mlpack.
#include <mlpack/core.hpp>

/**
 * Set the memory state of the given Armadillo object.
 */
template<typename T>
void SetMemState(T& t, int state)
{
  const_cast<arma::uhword&>(t.mem_state) = state;
}

#endif
