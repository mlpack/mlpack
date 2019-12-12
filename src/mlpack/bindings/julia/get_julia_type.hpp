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
inline std::string GetJuliaType(
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
{
  return "unknown_"; // This will cause an error most likely...
}

template<>
inline std::string GetJuliaType<bool>(
    const typename std::enable_if<!util::IsStdVector<bool>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<bool>::value>::type*,
    const typename std::enable_if<!std::is_same<bool,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<bool>::value>::type*)
{
  return "Bool";
}

template<>
inline std::string GetJuliaType<int>(
    const typename std::enable_if<!util::IsStdVector<int>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<int>::value>::type*,
    const typename std::enable_if<!std::is_same<int,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<int>::value>::type*)
{
  return "Int";
}

template<>
inline std::string GetJuliaType<size_t>(
    const typename std::enable_if<!util::IsStdVector<size_t>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<size_t>::value>::type*,
    const typename std::enable_if<!std::is_same<size_t,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<size_t>::value>::type*)
{
  return "UInt";
}

template<>
inline std::string GetJuliaType<double>(
    const typename std::enable_if<!util::IsStdVector<double>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<double>::value>::type*,
    const typename std::enable_if<!std::is_same<double,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<double>::value>::type*)
{
  // I suppose on some systems this may not be 64 bit.
  return "Float64";
}

template<>
inline std::string GetJuliaType<std::string>(
    const typename std::enable_if<
        !util::IsStdVector<std::string>::value>::type*,
    const typename std::enable_if<
        !arma::is_arma_type<std::string>::value>::type*,
    const typename std::enable_if<!std::is_same<std::string,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<
        !data::HasSerialize<std::string>::value>::type*)
{
  return "String";
}

template<typename T>
inline std::string GetJuliaType(
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0)
{
  return "Vector{" + GetJuliaType<typename T::value_type>() + "}";
}

template<typename T>
inline std::string GetJuliaType(
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  // size_t matrices are special: we want to represent them in Julia as
  // Array{Int64, X} not UInt64 because Julia displays UInt64s strangely.
  if (std::is_same<typename T::elem_type, size_t>::value)
    return std::string("Array{Int64, ") + (T::is_col || T::is_row ? "1" : "2")
        + "}";
  else
    return "Array{" + GetJuliaType<typename T::elem_type>() + ", "
        + (T::is_col || T::is_row ? "1" : "2") + "}";
}

template<typename T>
inline std::string GetJuliaType(
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  return "Tuple{Array{Bool, 1}, Array{Float64, 2}}";
}

// for serializable types
template<typename T>
inline std::string GetJuliaType(
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
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
