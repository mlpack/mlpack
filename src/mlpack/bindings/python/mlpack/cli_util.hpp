/**
 * @file cli_util.hpp
 * @author Ryan Curtin
 *
 * Simple function to work around Cython's lack of support for lvalue
 * references.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_CYTHON_CLI_UTIL_HPP
#define MLPACK_BINDINGS_PYTHON_CYTHON_CLI_UTIL_HPP

#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>

namespace mlpack {
namespace util {

/**
 * Set the parameter to the given value.
 *
 * This function exists to work around Cython's lack of support for lvalue
 * references.
 *
 * @param identifier Name of parameter.
 * @param value Value to set parameter to.
 */
template<typename T>
inline void SetParam(const std::string& identifier, const T& value)
{
  CLI::GetParam<T>(identifier) = value;
}

/**
 * Set the parameter (which is a matrix/DatasetInfo tuple) to the given value.
 */
template<typename T>
inline void SetParamWithInfo(const std::string& identifier,
                             const T& matrix,
                             const bool* dims)
{
  typedef typename std::tuple<data::DatasetInfo, T> TupleType;
  typedef typename T::elem_type eT;

  // The true type of the parameter is std::tuple<T, DatasetInfo>.
  std::get<1>(CLI::GetParam<TupleType>(identifier)) = matrix;
  data::DatasetInfo& di = std::get<0>(CLI::GetParam<TupleType>(identifier));
  di = data::DatasetInfo(matrix.n_rows);

  bool hasCategoricals = false;
  for (size_t i = 0; i < matrix.n_rows; ++i)
  {
    if (dims[i])
    {
      di.Type(i) = data::Datatype::categorical;
      hasCategoricals = true;
    }
  }

  // Do we need to find how many categories we have?
  if (hasCategoricals)
  {
    arma::vec maxs = arma::max(matrix, 1);

    for (size_t i = 0; i < matrix.n_rows; ++i)
    {
      if (dims[i])
      {
        // Map the right number of objects.
        for (size_t j = 0; j < (size_t) maxs[i]; ++j)
        {
          std::ostringstream oss;
          oss << j;
          di.MapString<eT>(oss.str(), i);
        }
      }
    }
  }
}

/**
 * Return the matrix part of a matrix + dataset info parameter.
 */
template<typename T>
T& GetParamWithInfo(const std::string& paramName)
{
  // T will be the Armadillo type.
  typedef std::tuple<data::DatasetInfo, T> TupleType;
  return std::get<1>(CLI::GetParam<TupleType>(paramName));
}

/**
 * Turn verbose output on.
 */
inline void EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

} // namespace util
} // namespace mlpack

#endif
