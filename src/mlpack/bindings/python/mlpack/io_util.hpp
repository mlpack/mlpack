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
#include <iostream>
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
inline void SetParam(const std::string& identifier, T& value)
{
  IO::GetParam<T>(identifier) = std::move(value);
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
inline void SetParamPtr(const std::string& identifier,
                        T* value,
                        const bool copy)
{
  IO::GetParam<T*>(identifier) = copy ? new T(*value) : value;
}

/**
 * Set the parameter (which is a matrix/DatasetInfo tuple) to the given value.
 */
template<typename T>
inline void SetParamWithInfo(const std::string& identifier,
                             T& matrix,
                             const bool* dims)
{
  typedef typename std::tuple<data::DatasetInfo, T> TupleType;
  typedef typename T::elem_type eT;

  // The true type of the parameter is std::tuple<T, DatasetInfo>.
  const size_t dimensions = matrix.n_rows;
  std::get<1>(IO::GetParam<TupleType>(identifier)) = std::move(matrix);
  data::DatasetInfo& di = std::get<0>(IO::GetParam<TupleType>(identifier));
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
        std::get<1>(IO::GetParam<TupleType>(identifier)), 1);

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
T* GetParamPtr(const std::string& paramName)
{
  return IO::GetParam<T*>(paramName);
}

/**
 * Return the matrix part of a matrix + dataset info parameter.
 */
template<typename T>
T& GetParamWithInfo(const std::string& paramName)
{
  // T will be the Armadillo type.
  typedef std::tuple<data::DatasetInfo, T> TupleType;
  return std::get<1>(IO::GetParam<TupleType>(paramName));
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
  IO::GetSingleton().timer.Reset();
}

/**
 * Enable timing.
 */
inline void EnableTimers()
{
  Timer::EnableTiming();
}

/**
 * Sanity Check.
 */
void SanityCheck()
{
  std::map<std::string, util::ParamData>::iterator itr;
  for (itr = IO::Parameters().begin(); itr != IO::Parameters().end(); ++itr)
  {
    std::string paramName = itr->first;
    std::string paramType = itr->second.cppType;
    if (paramType == "arma::mat")
    {
      if (IO::GetParam<arma::Mat<double>>(paramName).has_nan())
        Log::Fatal << "The input " << paramName << " has nan values." << std::endl;
    }
    else if (paramType == "arma::Mat<size_t>")
    {
      if (IO::GetParam<arma::Mat<size_t>>(paramName).has_nan())
        Log::Fatal << "The input " << paramName << " has nan values." << std::endl;
    }
    else if (paramType == "arma::colvec")
    {
      if (IO::GetParam<arma::Col<double>>(paramName).has_nan())
        Log::Fatal << "The input " << paramName << " has nan values." << std::endl;
    }
    else if (paramType == "arma::Col<size_t>")
    {
      if (IO::GetParam<arma::Col<size_t>>(paramName).has_nan())
        Log::Fatal << "The input " << paramName << " has nan values." << std::endl;
    }
    else if (paramType == "arma::rowvec")
    {
      if (IO::GetParam<arma::Row<double>>(paramName).has_nan())
        Log::Fatal << "The input " << paramName << " has nan values." << std::endl;
    }
    else if (paramType == "arma::Row<size_t>")
    {
      if (IO::GetParam<arma::Row<size_t>>(paramName).has_nan())
        Log::Fatal << "The input " << paramName << " has nan values." << std::endl;
    }
    else if (paramType == "std::tuple<data::DatasetInfo, arma::mat>")
    {
      if (std::get<1>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName)).has_nan())
        Log::Fatal << "The input " << paramName << " has nan values." << std::endl;
    }
  }
}

} // namespace util
} // namespace mlpack

#endif
