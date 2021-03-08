/**
 * @file bindings/R/print_type_doc_impl.hpp
 * @author Yashwant Singh Parihar
 *
 * Print documentation for a given type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_PRINT_TYPE_DOC_IMPL_HPP
#define MLPACK_BINDINGS_R_PRINT_TYPE_DOC_IMPL_HPP

#include "print_type_doc.hpp"

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Return a string representing the command-line type of an option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::disable_if<util::IsStdVector<T>>::type*,
    const typename boost::disable_if<data::HasSerialize<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  // A flag type.
  if (std::is_same<T, bool>::value)
  {
    return "A boolean flag option (i.e. `TRUE` or `FALSE`).";
  }
  // An integer.
  else if (std::is_same<T, int>::value)
  {
    return "An integer (i.e., `1`).";
  }
  // A floating point value.
  else if (std::is_same<T, double>::value)
  {
    return "A floating-point number (i.e., `0.5`).";
  }
  // A string.
  else if (std::is_same<T, std::string>::value)
  {
    return "A character string (i.e., `\"hello\"`).";
  }
  // Not sure what it is...
  else
  {
    throw std::invalid_argument("Unknown parameter type `" + data.cppType +
        "`.");
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
    return "A vector of integers; i.e., `c(0, 1, 2)`.";
  }
  else if (std::is_same<T, std::vector<std::string>>::value)
  {
    return "A vector of strings; i.e., `c(\"hello\", \"goodbye\")`.";
  }
  else
  {
    throw std::invalid_argument("Unknown vector type `" + data.cppType + "`.");
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
      return "A 1-d matrix-like containing `numeric` data (could be an "
          "`matrix` or a `data.frame` with one dimension of size 1).";
    }
    else
    {
      return "A 2-d matrix-like containing `numeric` data (could be an "
          "`matrix` or a `data.frame` or anything convertible to an "
          "2-d `matrix`).";
    }
  }
  else if (std::is_same<typename T::elem_type, size_t>::value)
  {
    if (T::is_col || T::is_row)
    {
      return "A 1-d matrix-like containing `integer` data (could be an "
          "`matrix` or a `data.frame` with one dimension of size 1).";
    }
    else
    {
      return "A 2-d matrix-like containing `integer` data (could be an "
          "`matrix` or a `data.frame` or anything convertible to an "
          "2-d `matrix`).";
    }
  }
  else
  {
    throw std::invalid_argument("Unknown matrix type `" + data.cppType + "`.");
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
  return "A 2-d array containing `numeric` data.  Like the regular 2-d matrices"
      ", this can be a `matrix`, or a `data.frame`. However, this type can also"
      " accept a `data.frame` that has columns of type `character`, `logical` "
      "or `factor`.  These values will be converted to `numeric` indices before"
      " being passed to mlpack, and then inside mlpack they will be properly "
      "treated as categorical variables, so there is no need to do one-hot "
      "encoding for this matrix type.";
}

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& /* data */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::enable_if<data::HasSerialize<T>>::type*)
{
  return "An mlpack model pointer.  `<Model>` refers to the type of model that "
      "is being stored, so, e.g., for `cf()`, the type will be `CFModel`. "
      "This type holds a pointer to C++ memory containing the mlpack model.  "
      "Note that this means the mlpack model itself cannot be easily inspected "
      "in R.  However, the pointer can be passed to subsequent calls to "
      "mlpack functions, and can be serialized and deserialized via either the "
      "`Serialize()` and `Unserialize()` functions.";
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
