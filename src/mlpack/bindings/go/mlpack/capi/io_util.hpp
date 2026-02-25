/**
 * @file bindings/go/mlpack/capi/io_util.hpp
 * @author Yasmine Dumouchel
 * @author Yashwant Singh
 *
 * Utility function for Go to set and get parameters to and from the IO.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_IO_UTIL_HPP
#define MLPACK_BINDINGS_GO_IO_UTIL_HPP

#include <mlpack/core/util/io.hpp>
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
inline void SetParam(util::Params& p,
                     const std::string& identifier,
                     T& value)
{
  p.Get<T>(identifier) = std::move(value);
}

/**
 * Set the parameter to the given value, given that the type is a pointer.
 *
 * @param identifier Name of parameter.
 * @param value Value to set parameter to.
 */
template<typename T>
inline void SetParamPtr(util::Params& p,
                        const std::string& identifier,
                        T* value)
{
  p.Get<T*>(identifier) = value;
}

/**
 * Return a pointer.  This function exists to work around Cython's seeming lack
 * of support for template pointer types.
 */
template<typename T>
T* GetParamPtr(util::Params& p, const std::string& paramName)
{
  return p.Get<T*>(paramName);
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

inline void ResetTimers()
{
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
