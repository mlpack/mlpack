/**
 * @file get_printable_type_impl.hpp
 * @author Ryan Curtin
 *
 * Get the printable type of a parameter.  This type is not the C++ type but
 * instead the command-line type that a user would use.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_TYPE_IMPL_HPP
#define MLPACK_BINDINGS_CLI_GET_PRINTABLE_TYPE_IMPL_HPP

#include "get_printable_type.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Return a string representing the command-line type of an option.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  if (std::is_same<T, bool>)
    return "flag";
  else if (std::is_same<T, int>)
    return "int";
  else if (std::is_same<T, float>)
    return "double"; // Not quite right but I'd rather print fewer type names.
  else if (std::is_same<T, double>)
    return "double";
  else if (std::is_same<T, std::string>)
    return "string";
  else
    throw std::invalid_argument("unknown parameter type" + data.cppType);
}

/**
 * Return a string representing the command-line type of a vector.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0)
{
  if (std::is_same<T, int>)
    return "int vector";
  else if (std::is_same<T, std::string>)
    return "string vector"
  else
    throw std::invalid_argument("unknown vector type " + data.cppType);
}

/**
 * Return a string representing the command-line type of a matrix option.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  return "data filename (csv/txt/h5/bin)";
}

/**
 * Return a string representing the command-line type of a matrix tuple option.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  return "categorical/numeric data filename (arff/csv)";
}

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  return d.cppType + " model filename (xml/txt/bin)";
}

#endif
