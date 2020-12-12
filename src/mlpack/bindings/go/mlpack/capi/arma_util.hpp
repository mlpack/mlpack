/**
 * @file bindings/go/mlpack/capi/arma_util.hpp
 * @author Ryan Curtin
 *
 * Utility function for Go to get memory pointer of an Armadillo Object.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_GONUM_ARMA_UTIL_HPP
#define MLPACK_BINDINGS_GO_GONUM_ARMA_UTIL_HPP

// Include Armadillo via mlpack.
#include <mlpack/core/util/io.hpp>
#include <mlpack/core.hpp>

namespace mlpack {

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
    arma::access::rw(m.mem_state) = 1;
    // With Armadillo 10 and newer, we must set `n_alloc` to 0 so that
    // Armadillo does not deallocate the memory.
    #if ARMA_VERSION_MAJOR >= 10
      arma::access::rw(m.n_alloc) = 0;
    #endif
    return m.memptr();
  }
}

} // namespace mlpack

#endif
