/**
 * @file bindings/python/mlpack/arma_util.hpp
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
  // If we just "released" the memory, so that the matrix does not own it, with
  // Armadillo 10 we must also ensure that the matrix does not deallocate the
  // memory by specifying `n_alloc = 0`.
  #if ARMA_VERSION_MAJOR >= 10
    const_cast<arma::uword&>(t.n_alloc) = 0;
  #endif
}

/**
 * Get the memory state of the given Armadillo object.
 */
template<typename T>
size_t GetMemState(T& t)
{
  // Fake the memory state if we are using preallocated memory---since we will
  // end up copying that memory, NumPy can own it.
  if (t.mem && t.n_elem <= arma::arma_config::mat_prealloc)
    return 0;

  return (size_t) t.mem_state;
}

/**
 * Return the matrix's allocated memory pointer, unless the matrix is using its
 * internal preallocated memory, in which case we copy that and return a
 * pointer to the memory we just made.
 */
template<typename T>
inline typename T::elem_type* GetMemory(T& m)
{
  if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
  {
    // We need to allocate new memory.
    typename T::elem_type* mem =
        arma::memory::acquire<typename T::elem_type>(m.n_elem);
    arma::arrayops::copy(mem, m.memptr(), m.n_elem);
    return mem;
  }
  else
  {
    return m.memptr();
  }
}

#endif
