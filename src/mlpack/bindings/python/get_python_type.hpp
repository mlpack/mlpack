/**
 * @file get_python_type.hpp
 * @author Ryan Curtin
 *
 * Template metaprogramming to return the string representation of the Python
 * type for a given Python binding parameter.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GET_PYTHON_TYPE_HPP
#define MLPACK_BINDINGS_PYTHON_GET_PYTHON_TYPE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

template<typename T>
inline std::string GetPythonType(
    const util::ParamData& /* d */,
    const typename boost::disable_if<IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0)
{
  return "unknown";
}

template<>
inline std::string GetPythonType<int>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<IsStdVector<int>>::type*,
    const typename boost::disable_if<data::HasSerialize<int>>::type*,
    const typename boost::disable_if<arma::is_arma_type<int>>::type*)
{
  return "int";
}

template<>
inline std::string GetPythonType<float>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<IsStdVector<float>>::type*,
    const typename boost::disable_if<data::HasSerialize<float>>::type*,
    const typename boost::disable_if<arma::is_arma_type<float>>::type*)
{
  return "float";
}

template<>
inline std::string GetPythonType<double>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<IsStdVector<double>>::type*,
    const typename boost::disable_if<data::HasSerialize<double>>::type*,
    const typename boost::disable_if<arma::is_arma_type<double>>::type*)
{
  return "double";
}

template<>
inline std::string GetPythonType<std::string>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<IsStdVector<std::string>>::type*,
    const typename boost::disable_if<data::HasSerialize<std::string>>::type*,
    const typename boost::disable_if<arma::is_arma_type<std::string>>::type*)
{
  return "string";
}

template<>
inline std::string GetPythonType<size_t>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<IsStdVector<size_t>>::type*,
    const typename boost::disable_if<data::HasSerialize<size_t>>::type*,
    const typename boost::disable_if<arma::is_arma_type<size_t>>::type*)
{
  return "size_t";
}

template<>
inline std::string GetPythonType<bool>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<IsStdVector<bool>>::type*,
    const typename boost::disable_if<data::HasSerialize<bool>>::type*,
    const typename boost::disable_if<arma::is_arma_type<bool>>::type*)
{
  return "bool";
}

template<typename T>
inline std::string GetPythonType(
    const util::ParamData& /* d */,
    const typename boost::enable_if<IsStdVector<T>>::type* = 0)
{
  return "array";
}

template<typename T>
inline std::string GetPythonType(
    const util::ParamData& d,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  return "arma.Mat[" + GetPythonType<typename T::elem_type>(d) + "]";
}

template<typename T>
inline std::string GetPythonType(
    const util::ParamData& d,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  return d.cppType;
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
