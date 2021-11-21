/**
 * @file bindings/julia/print_type_doc_impl.hpp
 * @author Ryan Curtin
 *
 * Print documentation for a given type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_TYPE_DOC_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_TYPE_DOC_IMPL_HPP

#include "print_type_doc.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

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
    return "A boolean flag option (`true` or `false`).";
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
    return "A vector of integers; i.e., `[0, 1, 2]`.";
  }
  else if (std::is_same<T, std::vector<std::string>>::value)
  {
    return "A vector of strings; i.e., `[\"hello\", \"goodbye\"]`.";
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
      return "A 1-d vector-like containing `Float64` data (could be an "
          "`Array{Float64, 1}`, an `Array{Float64, 2}` with one dimension "
          "of size 1, or anything convertible to `Array{Float64, 1}`.";
    }
    else
    {
      return "A 2-d matrix-like containing `Float64` data (could be an "
          "`Array{Float64, 2}` or a `DataFrame` or anything convertible to an "
          "`Array{Float64, 2}`).  It is expected that each row of the matrix "
          "corresponds to a data point, unless `points_are_rows` is set to "
          "`false` when calling mlpack bindings.";
    }
  }
  else if (std::is_same<typename T::elem_type, size_t>::value)
  {
    if (T::is_col || T::is_row)
    {
      return "A 1-d vector-like containing `Int` data (elements should be "
          "greater than or equal to 0).  Could be an `Array{Int, 1}`, an "
          "`Array{Int, 2}` with one dimension of size 1, or anything "
          "convertible to `Array{Int, 1}`.";
    }
    else
    {
      return "A 2-d matrix-like containing `Int` data (elements should be "
          "greater than or equal to 0).  Could be an `Array{Int, 2}` or a "
          "`DataFrame` or anything convertible to an `Array{Int, 2}`.  It is "
          "expected that each row of the matrix corresponds to a data point, "
          "unless `points_are_rows` is set to `false` when calling mlpack "
          "bindings.";
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
  return "A 2-d array containing `Float64` data along with a boolean array "
      "indicating which dimensions are categorical (represented by `true`) and "
      "which are numeric (represented by `false`).  The number of elements in "
      "the boolean array should be the same as the dimensionality of the data "
      "matrix.  Categorical dimensions should take integer values between 1 "
      "and the number of categories.  It is expected that each row of the "
      "matrix corresponds to a single data point, unless `points_are_rows` is "
      "set to `false` when calling mlpack bindings.";
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
  return "An mlpack model pointer.  `<Model>` refers to the type of model that "
      "is being stored, so, e.g., for `CF()`, the type will be `CFModel`. "
      "This type holds a pointer to C++ memory containing the mlpack model.  "
      "Note that this means the mlpack model itself cannot be easily inspected "
      "in Julia.  However, the pointer can be passed to subsequent calls to "
      "mlpack functions, and can be serialized and deserialized via either the "
      "`Serialization` package, or the `mlpack.serialize_bin()` and "
      "`mlpack.deserialize_bin()` functions.";
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
