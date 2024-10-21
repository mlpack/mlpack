/**
 * @file bindings/R/default_param_impl.hpp
 * @author Yashwant Singh Parihar
 *
 * Return the default value of a parameter, depending on its type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_DEFAULT_PARAM_IMPL_HPP
#define MLPACK_BINDINGS_R_DEFAULT_PARAM_IMPL_HPP

#include "default_param.hpp"

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Return the default value of an option.
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<!util::IsStdVector<T>::value>*,
    const std::enable_if_t<!data::HasSerialize<T>::value>*,
    const std::enable_if_t<!std::is_same_v<T, std::string>>*,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>*)
{
  std::ostringstream oss;
  if (std::is_same_v<T, bool>)
  {
    // If this is the verbose option, print the default that uses the global
    // package option.
    if (data.name == "verbose")
    {
      oss << "getOption(\"mlpack.verbose\", FALSE)";
    }
    else
    {
      oss << "FALSE";
    }
  }
  else
  {
    oss << std::any_cast<T>(data.value);
  }
  return oss.str();
}

/**
 * Return the default value of a vector option.
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& data,
    const std::enable_if_t<util::IsStdVector<T>::value>*)
{
  // Print each element in an array delimited by square brackets.
  std::ostringstream oss;
  const T& vector = std::any_cast<T>(data.value);
  oss << "c(";
  if (std::is_same_v<T, std::vector<std::string>>)
  {
    if (vector.size() > 0)
    {
      for (size_t i = 0; i < vector.size() - 1; ++i)
      {
        oss << "'" << vector[i] << "', ";
      }

      oss << "'" << vector[vector.size() - 1] << "'";
    }

    oss << ")";
  }
  else
  {
    if (vector.size() > 0)
    {
      for (size_t i = 0; i < vector.size() - 1; ++i)
      {
        oss << vector[i] << ", ";
      }

      oss << vector[vector.size() - 1];
    }

    oss << ")";
  }
  return oss.str();
}

/**
 * Return the default value of a string option.
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& data,
    const std::enable_if_t<std::is_same_v<T, std::string>>*)
{
  const std::string& s = *std::any_cast<std::string>(&data.value);
  return "\"" + s + "\"";
}

/**
 * Return the default value of a matrix option (this returns the default
 * filename, or '' if the default is no file).
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& /* data */,
    const std::enable_if_t<
        arma::is_arma_type<T>::value ||
        std::is_same_v<T, std::tuple<mlpack::data::DatasetInfo,
                                     arma::mat>>>* /* junk */)
{
  // Get the filename and return it, or return an empty string.
  if (std::is_same_v<T, arma::rowvec> ||
      std::is_same_v<T, arma::vec> ||
      std::is_same_v<T, arma::mat>)
  {
    return "matrix(numeric(), 0, 0)";
  }
  else if (std::is_same_v<T, arma::Row<size_t>> ||
      std::is_same_v<T, arma::Col<size_t>> ||
      std::is_same_v<T, arma::Mat<size_t>>)
  {
    return "matrix(integer(), 0, 0)";
  }
  else
  {
    return "matrix(numeric(), 0, 0)";
  }
}

/**
 * Return the default value of a model option (always "None").
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& /* data */,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<data::HasSerialize<T>::value>*)
{
  return "NA";
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
