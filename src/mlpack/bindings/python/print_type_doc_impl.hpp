/**
 * @file print_type_doc_impl.hpp
 * @author Ryan Curtin
 *
 * Print documentation for a given type.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_TYPE_DOC_IMPL_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_TYPE_DOC_IMPL_HPP

#include "print_type_doc.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Return a string representing the command-line type of an option.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& /* data */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::disable_if<util::IsStdVector<T>>::type*,
    const typename boost::disable_if<data::HasSerialize<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  // A flag type.
  if (std::is_same<T, bool>::value)
  {
    return "A boolean flag option.  If not specified, it is false; if "
        "specified, it is true.";
  }
  // An integer.
  else if (std::is_same<T, int>::value)
  {
    return "An integer (i.e., \"1\").";
  }
  // A floating point value.
  else if (std::is_same<T, double>::value)
  {
    return "A floating-point number (i.e., \"0.5\").";
  }
  // A string.
  else if (std::is_same<T, std::string>::value)
  {
    return "A character string (i.e., \"hello\").";
  }
  // Not sure what it is...
  else
  {
    throw std::invalid_argument("unknown parameter type");
  }
}

/**
 * Return a string representing the command-line type of a vector.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& /* data */,
    const typename std::enable_if<util::IsStdVector<T>::value>::type*)
{
  if (std::is_same<T, std::vector<int>>::value)
  {
    return "A vector of integers, separated by commas (i.e., \"1,2,3\").";
  }
  else if (std::is_same<T, std::vector<std::string>>::value)
  {
    return "A vector of strings, separated by commas (i.e., "
        "\"hello\",\"goodbye\").";
  }
  else
  {
    throw std::invalid_argument("unknown vector type");
  }
}

/**
 * Return a string representing the command-line type of a matrix option.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& /* data */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*)
{
  return "A data matrix file.  This can take many types and we need better "
      "documentation here.";
}

/**
 * Return a string representing the command-line type of a matrix tuple option.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& /* data */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "A data matrix that can contain categorical data.";
}

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string PrintTypeDoc(
    const util::ParamData& /* data */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::enable_if<data::HasSerialize<T>>::type*)
{
  return "A model type.";
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
