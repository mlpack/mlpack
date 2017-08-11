/**
 * @file get_numpy_type_char.hpp
 * @author Ryan Curtin
 *
 * Given a matrix type, return the letter we should append to get the right
 * arma_numpy method call.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GET_NUMPY_TYPE_CHAR_HPP
#define MLPACK_BINDINGS_PYTHON_GET_NUMPY_TYPE_CHAR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

template<typename T>
inline std::string GetNumpyTypeChar()
{
  return "?";
}

// size_t = s.
template<>
inline std::string GetNumpyTypeChar<arma::Mat<size_t>>()
{
  return "s";
}

template<>
inline std::string GetNumpyTypeChar<arma::Col<size_t>>()
{
  return "s";
}

template<>
inline std::string GetNumpyTypeChar<arma::Row<size_t>>()
{
  return "s";
}

// double = d.
template<>
inline std::string GetNumpyTypeChar<arma::mat>()
{
  return "d";
}

template<>
inline std::string GetNumpyTypeChar<arma::vec>()
{
  return "d";
}

template<>
inline std::string GetNumpyTypeChar<arma::rowvec>()
{
  return "d";
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
