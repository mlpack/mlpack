/**
 * @file bindings/cli/get_allocated_memory.hpp
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
#ifndef MLPACK_BINDINGS_CLI_GET_ALLOCATED_MEMORY_HPP
#define MLPACK_BINDINGS_CLI_GET_ALLOCATED_MEMORY_HPP

#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

template<typename T>
void* GetAllocatedMemory(
    util::ParamData& /* d */,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0)
{
  return NULL;
}

template<typename T>
void* GetAllocatedMemory(
    util::ParamData& /* d */,
    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0)
{
  return NULL;
}

template<typename T>
void* GetAllocatedMemory(
    util::ParamData& d,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  // Here we have a model, which is a tuple, and we need the address of the
  // memory.
  using TupleType = std::tuple<T*, std::string>;
  return std::get<0>(*std::any_cast<TupleType>(&d.value));
}

template<typename T>
void GetAllocatedMemory(util::ParamData& d,
                        const void* /* input */,
                        void* output)
{
  *((void**) output) = GetAllocatedMemory<std::remove_pointer_t<T>>(d);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
