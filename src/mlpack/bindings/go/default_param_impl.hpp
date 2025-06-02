/**
 * @file bindings/go/default_param_impl.hpp
 * @author Yashwant Singh
 *
 * Return the default value of a parameter, depending on its type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_DEFAULT_PARAM_IMPL_HPP
#define MLPACK_BINDINGS_GO_DEFAULT_PARAM_IMPL_HPP

#include "default_param.hpp"

namespace mlpack {
namespace bindings {
namespace go {

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
    oss << "false";
  else
    oss << std::any_cast<T>(data.value);

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
  if (std::is_same_v<T, std::vector<std::string>>)
  {
    oss << "[]string{";
    if (vector.size() > 0)
    {
      for (size_t i = 0; i < vector.size() - 1; ++i)
      {
        oss << "\"" << vector[i] << "\", ";
      }

      oss << "\"" << vector[vector.size() - 1] << "\"";
    }

    oss << "}";
  }
  else if (std::is_same_v<T, std::vector<int>>)
  {
    oss << "[]int{";
    if (vector.size() > 0)
    {
      for (size_t i = 0; i < vector.size() - 1; ++i)
      {
        oss << vector[i] << ", ";
      }

      oss << vector[vector.size() - 1];
    }

    oss << "}";
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
 * Return the default value of a matrix option.
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
  if (std::is_same_v<T, arma::rowvec> || std::is_same_v<T, arma::vec>)
  {
    return "mat.NewDense(1, 1, nil)";
  }
  else if (std::is_same_v<T, arma::Col<size_t>> ||
           std::is_same_v<T, arma::Row<size_t>>)
  {
    return "mat.NewDense(1, 1, nil)";
  }
  else if (std::is_same_v<T, arma::Mat<size_t>>)
  {
    return "mat.NewDense(1, 1, nil)";
  }
  else
  {
    return "mat.NewDense(1, 1, nil)";
  }
}

/**
 * Return the default value of a model option (always "nil").
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& /* data */,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<data::HasSerialize<T>::value>*)
{
  return "nil";
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
