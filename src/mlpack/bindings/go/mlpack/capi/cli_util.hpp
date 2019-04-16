/**
 * @file cli_util.hpp
 * @author Yasmine Dumouchel
 * @author Yashwant Singh
 *
 * Utility function for Go to set and get parameters to and from the CLI.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_CLI_UTIL_HPP
#define MLPACK_BINDINGS_GO_CLI_UTIL_HPP

#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>

namespace mlpack {
namespace util {

/**
 * Set the parameter to the given value.
 *
 * @param identifier Name of parameter.
 * @param value Value to set parameter to.
 */
template<typename T>
inline void SetParam(const std::string& identifier, T& value)
{
  CLI::GetParam<T>(identifier) = std::move(value);
}

/**
 * Set the parameter to the given value, given that the type is a pointer.
 *
 * @param identifier Name of parameter.
 * @param value Value to set parameter to.
 * @param copy Whether or not the object should be copied.
 */
template<typename T>
inline void SetParamPtr(const std::string& identifier,
                        T* value)
{
  CLI::GetParam<T*>(identifier) = value;
}

/**
 * Return a pointer.  This function exists to work around Cython's seeming lack
 * of support for template pointer types.
 */
template<typename T>
T* GetParamPtr(const std::string& paramName)
{
  return CLI::GetParam<T*>(paramName);
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
  CLI::GetSingleton().timer.Reset();
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
