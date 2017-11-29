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

/**
 * Return the matrix's allocated memory pointer, unless the matrix is using its
 * internal preallocated memory, in which case we copy that and return a
 * pointer to the memory we just made.
 */
inline double* GetMemory(arma::mat& m)
{
  if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
  {
    // We need to allocate new memory.
    double* mem = arma::memory::acquire<double>(m.n_elem);
    arma::arrayops::copy(mem, m.memptr(), m.n_elem);
    return mem;
  }
  else
  {
    return m.memptr();
  }
}

/**
 * Return the matrix's allocated memory pointer, unless the matrix is using its
 * internal preallocated memory, in which case we copy that and return a
 * pointer to the memory we just made.
 */
inline double* GetMemory(arma::vec& m)
{
  if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
  {
    // We need to allocate new memory.
    double* mem = arma::memory::acquire<double>(m.n_elem);
    arma::arrayops::copy(mem, m.memptr(), m.n_elem);
    return mem;
  }
  else
  {
    return m.memptr();
  }
}

/**
 * Return the matrix's allocated memory pointer, unless the matrix is using its
 * internal preallocated memory, in which case we copy that and return a
 * pointer to the memory we just made.
 */
inline double* GetMemory(arma::rowvec& m)
{
  if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
  {
    // We need to allocate new memory.
    double* mem = arma::memory::acquire<double>(m.n_elem);
    arma::arrayops::copy(mem, m.memptr(), m.n_elem);
    return mem;
  }
  else
  {
    return m.memptr();
  }
}

/**
 * Return the matrix's allocated memory pointer, unless the matrix is using its
 * internal preallocated memory, in which case we copy that and return a
 * pointer to the memory we just made.
 */
inline size_t* GetMemory(arma::Mat<size_t>& m)
{
  if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
  {
    // We need to allocate new memory.
    size_t* mem = arma::memory::acquire<size_t>(m.n_elem);
    arma::arrayops::copy(mem, m.memptr(), m.n_elem);
    return mem;
  }
  else
  {
    return m.memptr();
  }
}

/**
 * Return the matrix's allocated memory pointer, unless the matrix is using its
 * internal preallocated memory, in which case we copy that and return a
 * pointer to the memory we just made.
 */
inline size_t* GetMemory(arma::Col<size_t>& m)
{
  if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
  {
    // We need to allocate new memory.
    size_t* mem = arma::memory::acquire<size_t>(m.n_elem);
    arma::arrayops::copy(mem, m.memptr(), m.n_elem);
    return mem;
  }
  else
  {
    return m.memptr();
  }
}

/**
 * Return the matrix's allocated memory pointer, unless the matrix is using its
 * internal preallocated memory, in which case we copy that and return a
 * pointer to the memory we just made.
 */
inline size_t* GetMemory(arma::Row<size_t>& m)
{
  if (m.mem && m.n_elem <= arma::arma_config::mat_prealloc)
  {
    // We need to allocate new memory.
    size_t* mem = arma::memory::acquire<size_t>(m.n_elem);
    arma::arrayops::copy(mem, m.memptr(), m.n_elem);
    return mem;
  }
  else
  {
    return m.memptr();
  }
}

#endif
