/**
 * @file bindings/python/is_serializable.hpp
 * @author Nippun Sharma
 *
 * Check if parameter is serializable.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_IS_SERIALIZABLE_HPP
#define MLPACK_BINDINGS_PYTHON_IS_SERIALIZABLE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

template<typename T>
inline bool IsSerializable(
    util::ParamData& /* d */,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
{
  return false;
}

template<typename T>
inline bool IsSerializable(
    util::ParamData& /* d */,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  return true;
}

template<typename T>
void IsSerializable(util::ParamData& data,
                    const void* /* input */,
                    void* output)
{
  *((bool*) output) =
      IsSerializable<typename std::remove_pointer<T>::type>(data);
}

} // python
} // bindings
} // mlpack

#endif
