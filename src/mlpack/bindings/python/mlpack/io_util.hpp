/**
 * @file bindings/python/mlpack/io_util.hpp
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
#ifndef MLPACK_BINDINGS_PYTHON_CYTHON_IO_UTIL_HPP
#define MLPACK_BINDINGS_PYTHON_CYTHON_IO_UTIL_HPP

#include <mlpack/core/util/io.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>

namespace mlpack {
namespace util {

// Utility functions to correctly handle transposed Armadillo matrices.
template<typename T>
inline void TransposeIfNeeded(
    const std::string& identifier,
    T& value,
    bool transpose)
{
  // No transpose needed for non-matrices.
  return;
}

inline void TransposeIfNeeded(
    const std::string& identifier,
    arma::mat& value,
    bool transpose)
{
  if (transpose)
  {
    arma::inplace_trans(value);
  }
}

/**
 * Set the parameter to the given value.
 *
 * This function exists to work around Cython's lack of support for lvalue
 * references.
 *
 * @param params Parameters object to use.
 * @param identifier Name of parameter.
 * @param value Value to set parameter to.
 * @param transpose If true, and if T is a matrix type, the matrix will be
 *     transposed in-place.
 */
template<typename T>
inline void SetParam(util::Params& params,
                     const std::string& identifier,
                     T& value,
                     bool transpose = false)
{
  TransposeIfNeeded(identifier, value, transpose);
  params.Get<T>(identifier) = std::move(value);
}

/**
 * Set the parameter to the given value, given that the type is a pointer.
 *
 * This function exists to work around both Cython's lack of support for lvalue
 * references and also its seeming lack of support for template pointer types.
 *
 * @param identifier Name of parameter.
 * @param value Value to set parameter to.
 * @param copy Whether or not the object should be copied.
 */
template<typename T>
inline void SetParamPtr(util::Params& params,
                        const std::string& identifier,
                        T* value,
                        const bool copy)
{
  params.Get<T*>(identifier) = copy ? new T(*value) : value;
}

/**
 * Set the parameter (which is a matrix/DatasetInfo tuple) to the given value.
 */
template<typename T>
inline void SetParamWithInfo(util::Params& params,
                             const std::string& identifier,
                             T& matrix,
                             const bool* dims)
{
  typedef typename std::tuple<data::DatasetInfo, T> TupleType;
  typedef typename T::elem_type eT;

  // The true type of the parameter is std::tuple<T, DatasetInfo>.
  const size_t dimensions = matrix.n_rows;
  std::get<1>(params.Get<TupleType>(identifier)) = std::move(matrix);
  data::DatasetInfo& di = std::get<0>(params.Get<TupleType>(identifier));
  di = data::DatasetInfo(dimensions);

  bool hasCategoricals = false;
  for (size_t i = 0; i < dimensions; ++i)
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
    arma::vec maxs = arma::max(
        std::get<1>(params.Get<TupleType>(identifier)), 1) + 1;

    for (size_t i = 0; i < dimensions; ++i)
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
 * Return a pointer.  This function exists to work around Cython's seeming lack
 * of support for template pointer types.
 */
template<typename T>
T* GetParamPtr(util::Params& params,
               const std::string& paramName)
{
  return params.Get<T*>(paramName);
}

/**
 * Return the matrix part of a matrix + dataset info parameter.
 */
template<typename T>
T& GetParamWithInfo(util::Params& params,
                    const std::string& paramName)
{
  // T will be the Armadillo type.
  typedef std::tuple<data::DatasetInfo, T> TupleType;
  return std::get<1>(params.Get<TupleType>(paramName));
}

/**
 * Turn verbose output on.
 */
inline void EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

/**
 * Turn verbose output off.
 */
inline void DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

/**
 * Disable backtraces.
 */
inline void DisableBacktrace()
{
  Log::Fatal.backtrace = false;
}

/**
 * Reset the status of all timers.
 */
inline void ResetTimers()
{
  // Just get a new object---removes all old timers.
  Timer::ResetAll();
}

/**
 * Enable timing.
 */
inline void EnableTimers()
{
  Timer::EnableTiming();
}

} // namespace util
} // namespace mlpack

#endif
