/**
 * @file get_numpy_type.hpp
 * @author Ryan Curtin
 *
 * Given a C++ type, return the Python numpy dtype associated with that type.
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
