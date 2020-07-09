/**
 * @file bindings/go/mlpack/capi/io_util.cpp
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
#include <mlpack/bindings/go/mlpack/capi/io_util.h>
#include "io_util.hpp"
#include <mlpack/core/util/io.hpp>

namespace mlpack {

extern "C" {

/**
 * Set the double parameter to the given value.
 */
void mlpackSetParamDouble(const char* identifier, double value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the int parameter to the given value.
 */
void mlpackSetParamInt(const char* identifier, int value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the float parameter to the given value.
 */
void mlpackSetParamFloat(const char* identifier, float value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the bool parameter to the given value.
 */
void mlpackSetParamBool(const char* identifier, bool value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the string parameter to the given value.
 */
void mlpackSetParamString(const char* identifier, const char* value)
{
  IO::GetParam<std::string>(identifier) = value;
}

/**
 * Set the int vector parameter to the given value.
 */
void mlpackSetParamVectorInt(const char* identifier,
                             const long long* ints,
                             const size_t length)
{
  // Create a std::vector<int> object; unfortunately this requires copying the
  // vector elements.
  std::vector<int> vec(length);
  for (size_t i = 0; i < length; ++i)
    vec[i] = ints[i];

  IO::GetParam<std::vector<int>>(identifier) = std::move(vec);
  IO::SetPassed(identifier);
}

/**
 * Call IO::SetParam<std::vector<std::string>>() to set the length.
 */
void mlpackSetParamVectorStrLen(const char* identifier,
                                const size_t length)
{
  IO::GetParam<std::vector<std::string>>(identifier).clear();
  IO::GetParam<std::vector<std::string>>(identifier).resize(length);
  IO::SetPassed(identifier);
}

/**
 * Set the string vector parameter to the given value.
 */
void mlpackSetParamVectorStr(const char* identifier,
                             const char* str,
                             const size_t element)
{
  IO::GetParam<std::vector<std::string>>(identifier)[element] =
      std::string(str);
}

/**
 * Set the parameter to the given value, given that the type is a pointer.
 */
void mlpackSetParamPtr(const char* identifier,
                       const double* ptr)
{
  util::SetParamPtr(identifier, ptr);
}

/**
 * Check if IO has a specified parameter.
 */
bool mlpackHasParam(const char* identifier)
{
  return IO::HasParam(identifier);
}

/**
 * Get the string parameter associated with specified identifier.
 */
const char* mlpackGetParamString(const char* identifier)
{
  return IO::GetParam<std::string>(identifier).c_str();
}

/**
 * Get the double parameter associated with specified identifier.
 */
double mlpackGetParamDouble(const char* identifier)
{
  return IO::GetParam<double>(identifier);
}

/**
 * Get the int parameter associated with specified identifier.
 */
int mlpackGetParamInt(const char* identifier)
{
  return IO::GetParam<int>(identifier);
}

/**
 * Get the bool parameter associated with specified identifier.
 */
bool mlpackGetParamBool(const char* identifier)
{
  return IO::GetParam<bool>(identifier);
}

/**
 * Get the vector<int> parameter associated with specified identifier.
 */
void* mlpackGetVecIntPtr(const char* identifier)
{
  const size_t size = mlpackVecIntSize(identifier);
  long long* ints = new long long[size];

  for (size_t i = 0; i < size; i++)
    ints[i] = IO::GetParam<std::vector<int>>(identifier)[i];

  return ints;
}

/**
 * Get the vector<string> parameter associated with specified identifier.
 */
const char* mlpackGetVecStringPtr(const char* identifier, const size_t i)
{
  return IO::GetParam<std::vector<std::string>>(identifier)[i].c_str();
}

/**
 * Get the vector<int> parameter's size.
 */
int mlpackVecIntSize(const char* identifier)
{
  return IO::GetParam<std::vector<int>>(identifier).size();
}

/**
 * Get the vector<string> parameter's size.
 */
int mlpackVecStringSize(const char* identifier)
{
  return IO::GetParam<std::vector<std::string>>(identifier).size();
}

/**
 * Set parameter as passed.
 */
void mlpackSetPassed(const char* name)
{
  IO::SetPassed(name);
}

/**
 * Reset the status of all timers.
 */
void mlpackResetTimers()
{
  IO::GetSingleton().timer.Reset();
}

/**
 * Enable timing.
 */
void mlpackEnableTimers()
{
  Timer::EnableTiming();
}

/**
 * Disable backtraces.
 */
void mlpackDisableBacktrace()
{
  Log::Fatal.backtrace = false;
}

/**
 * Turn verbose output on.
 */
void mlpackEnableVerbose()
{
  Log::Info.ignoreInput = false;
}

/**
 * Turn verbose output off.
 */
void mlpackDisableVerbose()
{
  Log::Info.ignoreInput = true;
}

/**
 * Clear settings.
 */
void mlpackClearSettings()
{
  IO::ClearSettings();
}

/**
 * Restore Settings.
 */
void mlpackRestoreSettings(const char* name)
{
  IO::RestoreSettings(name);
}

} // extern C

} // namespace mlpack
