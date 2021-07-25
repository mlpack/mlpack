/**
 * @file bindings/python/print_type_doc_impl.hpp
 * @author Ryan Curtin
 *
 * Print documentation for a given type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
    util::ParamData& data,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  // A flag type.
  if (std::is_same<T, bool>::value)
  {
    return "A boolean flag option (True or False).";
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
    throw std::invalid_argument("unknown parameter type " + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a vector.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type*)
{
  if (std::is_same<T, std::vector<int>>::value)
  {
    return "A list of integers; i.e., [0, 1, 2].";
  }
  else if (std::is_same<T, std::vector<std::string>>::value)
  {
    return "A list of strings; i.e., [\"hello\", \"goodbye\"].";
  }
  else
  {
    throw std::invalid_argument("unknown vector type " + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a matrix option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*)
{
  if (std::is_same<typename T::elem_type, double>::value)
  {
    if (T::is_col || T::is_row)
    {
      return "A 1-d arraylike containing data.  This can be a 2-d matrix where "
          "one dimension has size 1, or it can also be a list, a numpy 1-d "
          "ndarray, or a 1-d pandas DataFrame.  If the dtype is not already "
          "float64, it will be converted.";
    }
    else
    {
      return "A 2-d arraylike containing data.  This can be a list of lists, a "
          "numpy ndarray, or a pandas DataFrame.  If the dtype is not already "
          "float64, it will be converted.";
    }
  }
  else if (std::is_same<typename T::elem_type, size_t>::value)
  {
    if (T::is_col || T::is_row)
    {
      return "A 1-d arraylike containing data with a uint64 dtype.  This can be"
          " a 2-d matrix where one dimension has size 1, or it can also be a "
          "list, a numpy 1-d ndarray, or a 1-d pandas DataFrame.  If the dtype "
          "is not already uint64, it will be converted.";
    }
    else
    {
      return "A 2-d arraylike containing data with a uint64 dtype.  This can "
          "be a list of lists, a numpy ndarray, or a pandas DataFrame.  If the "
          "dtype is not already uint64, it will be converted.";
    }
  }
  else
  {
    throw std::invalid_argument("unknown matrix type " + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a matrix tuple option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& /* data */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "A 2-d arraylike containing data.  Like the regular 2-d matrices, this"
      " can be a list of lists, a numpy ndarray, or a pandas DataFrame. "
      "However, this type can also accept a pandas DataFrame that has columns "
      "of type 'CategoricalDtype'.  These categorical values will be converted "
      "to numeric indices before being passed to mlpack, and then inside mlpack"
      " they will be properly treated as categorical variables, so there is no "
      "need to do one-hot encoding for this matrix type.  If the dtype of the "
      "given matrix is not already float64, it will be converted.";
}

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& /* data */,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<data::HasSerialize<T>::value>::type*)
{
  return "An mlpack model pointer.  This type can be pickled to or from disk, "
      "and internally holds a pointer to C++ memory containing the mlpack "
      "model.  This model pointer has 2 methods with which the parameters "
      "of the model can be inspected as well as changed through Python.  "
      "The `get_cpp_params()` method returns a python ordered dictionary that "
      "contains all the parameters of the model.  These parameters can "
      "be inspected and changed.  To set new parameters for a model, "
      "pass the modified dictionary (without deleting any keys) to the "
      "`set_cpp_params()` method.";
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
