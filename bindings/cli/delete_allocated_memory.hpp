/**
 * @file bindings/cli/delete_allocated_memory.hpp
 * @author Ryan Curtin
 *
 * If any memory has been allocated by the parameter, delete it.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_DELETE_ALLOCATED_MEMORY_HPP
#define MLPACK_BINDINGS_CLI_DELETE_ALLOCATED_MEMORY_HPP

#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

template<typename T>
void DeleteAllocatedMemoryImpl(
    util::ParamData& /* d */,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0)
{
  // Do nothing.
}

template<typename T>
void DeleteAllocatedMemoryImpl(
    util::ParamData& /* d */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  // Do nothing.
}

template<typename T>
void DeleteAllocatedMemoryImpl(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // Delete the allocated memory (hopefully we actually own it).
  typedef std::tuple<T*, std::string> TupleType;
  delete std::get<0>(*std::any_cast<TupleType>(&d.value));
}

template<typename T>
void DeleteAllocatedMemory(
    util::ParamData& d,
    const void* /* input */,
    void* /* output */)
{
  DeleteAllocatedMemoryImpl<typename std::remove_pointer<T>::type>(d);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
