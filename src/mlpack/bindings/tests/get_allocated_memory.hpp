/**
 * @file bindings/tests/get_allocated_memory.hpp
 * @author Ryan Curtin
 *
 * If the parameter has a type that may need to be deleted, return the address
 * of that object.  Otherwise return NULL.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_IO_GET_ALLOCATED_MEMORY_HPP
#define MLPACK_BINDINGS_IO_GET_ALLOCATED_MEMORY_HPP

#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace tests {

template<typename T>
void* GetAllocatedMemory(
    util::ParamData& /* d */,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0)
{
  return NULL;
}

template<typename T>
void* GetAllocatedMemory(
    util::ParamData& d,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  return (*MLPACK_ANY_CAST<T>(&d.value)).memptr();
}

template<typename T>
void* GetAllocatedMemory(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // Here we have a model; return its memory location.
  return *MLPACK_ANY_CAST<T*>(&d.value);
}

template<typename T>
void GetAllocatedMemory(util::ParamData& d,
                        const void* /* input */,
                        void* output)
{
  *((void**) output) =
      GetAllocatedMemory<typename std::remove_pointer<T>::type>(d);
}

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
