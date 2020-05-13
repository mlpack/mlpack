/**
 * @file bindings/python/get_numpy_type.hpp
 * @author Ryan Curtin
 *
 * Given a C++ type, return the Python numpy dtype associated with that type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GET_NUMPY_TYPE_HPP
#define MLPACK_BINDINGS_PYTHON_GET_NUMPY_TYPE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

template<typename T>
inline std::string GetNumpyType()
{
  return "unknown"; // Not sure...
}

template<>
inline std::string GetNumpyType<double>()
{
  return "np.double";
}

template<>
inline std::string GetNumpyType<size_t>()
{
  return "np.intp";
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
