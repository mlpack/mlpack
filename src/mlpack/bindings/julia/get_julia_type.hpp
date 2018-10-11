/**
 * @file get_julia_type.hpp
 * @author Ryan Curtin
 *
 * Get the Julia-named type of an mlpack C++ type.
 */
#ifndef MLPACK_BINDINGS_JULIA_GET_JULIA_TYPE_HPP
#define MLPACK_BINDINGS_JULIA_GET_JULIA_TYPE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

template<typename T>
inline std::string GetJuliaType()
{
  return "unknown_"; // This will cause an error most likely...
}

template<>
inline std::string GetJuliaType<bool>()
{
  return "Bool";
}

template<>
inline std::string GetJuliaType<int>()
{
  return "Int";
}

template<>
inline std::string GetJuliaType<double>()
{
  // I suppose on some systems this may not be 64 bit.
  return "Float64";
}

template<>
inline std::string GetJuliaType<std::string>()
{
  return "String";
}

template<>
inline std::string GetJuliaType<arma::mat>()
{
  return "Array{" + GetJuliaType<double>() + ", 2}";
}

template<>
inline std::string GetJuliaType<arma::Mat<size_t>>()
{
  return "Array{" + GetJuliaType<size_t>() + ", 2}";
}

template<>
inline std::string GetJuliaType<arma::vec>()
{
  return "Array{" + GetJuliaType<double>() + ", 1}";
}

template<>
inline std::string GetJuliaType<arma::Col<size_t>>()
{
  return "Array{" + GetJuliaType<size_t>() + ", 1}";
}

template<>
inline std::string GetJuliaType<arma::rowvec>()
{
  return "Array{" + GetJuliaType<double>() + ", 1}";
}

template<>
inline std::string GetJuliaType<arma::Row<size_t>>()
{
  return "Array{" + GetJuliaType<size_t>() + ", 1}";
}

// for serializable types
template<typename T>
inline std::string GetJuliaType(
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // Serializable types are just held as a pointer to nothing...
  return "Ptr{Nothing}";
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
