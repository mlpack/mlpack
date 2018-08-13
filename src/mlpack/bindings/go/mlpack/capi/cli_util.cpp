/**
 * @file cli_util.cpp
 * @author Yasmine Dumouchel
 *
 * Utility function for Go to set and get parameters to and from the CLI.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "cli_util.h"
#include "cli_util.hpp"
#include <mlpack/core/util/cli.hpp>

namespace mlpack {

extern "C" {

/**
 * Set the double parameter to the given value.
 */
void MLPACK_SetParamDouble(const char *identifier, double value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the int parameter to the given value.
 */
void MLPACK_SetParamInt(const char *identifier, int value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the float parameter to the given value.
 */
void MLPACK_SetParamFloat(const char *identifier, float value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the bool parameter to the given value.
 */
void MLPACK_SetParamBool(const char *identifier, bool value)
{
  util::SetParam(identifier, value);
}

/**
 * Set the string parameter to the given value.
 */
void MLPACK_SetParamString(const char *identifier, const char *value)
{
  std::string val;
  val.assign(value);
  util::SetParam(identifier, val);
}

/**
 * Set the parameter to the given value, given that the type is a pointer.
 */
void MLPACK_SetParamPtr(const char *identifier, const double *ptr, const bool copy)
{
  util::SetParamPtr(identifier, ptr, copy);
}

/**
 * Check if CLI has a specified parameter.
 */
bool MLPACK_HasParam(const char *identifier)
{
  return CLI::HasParam(identifier);
}

/**
 * Get the string parameter associated with specified identifier.
 */
char *MLPACK_GetParamString(const char *identifier)
{
  std::string val = CLI::GetParam<std::string>(identifier);
  char *cstr = const_cast<char*>(val.c_str());
  return cstr;
}

/**
 * Get the double parameter associated with specified identifier.
 */
double MLPACK_GetParamDouble(const char *identifier)
{
  double val = CLI::GetParam<double>(identifier);
  return val;
}

/**
 * Get the int parameter associated with specified identifier.
 */
int MLPACK_GetParamInt(const char *identifier)
{
  int val = CLI::GetParam<int>(identifier);
  return val;
}

/**
 * Get the bool parameter associated with specified identifier.
 */
bool MLPACK_GetParamBool(const char *identifier)
{
  bool val = CLI::GetParam<bool>(identifier);
  return val;
}

/**
 * Get the vector<int> parameter associated with specified identifier.
 */
void *MLPACK_GetVecIntPtr(const char *identifier)
{
  // std::vector<int> vec = CLI::GetParam<std::vector<int>>(identifier);
  // return vec.get_allocator();
}

/**
 * Get the vector<string> parameter associated with specified identifier.
 */
void *MLPACK_GetVecStringPtr(const char *identifier)
{
  // std::vector<std::string> vec = CLI::GetParam<std::vector<std::string>>(identifier);
  // return vec.get_allocator();
}

/**
 * Get the vector<int> parameter's size.
 */
int MLPACK_VecIntSize(const char *identifier)
{
  std::vector<int> output = CLI::GetParam<std::vector<int>>(identifier);
  return output.size();
}

/**
 * Get the vector<string> parameter's size.
 */
int MLPACK_VecStringSize(const char *identifier)
{
  std::vector<std::string> output = CLI::GetParam<std::vector<std::string>>(identifier);
  return output.size();
}

/**
 * Set parameter as passed.
 */
void MLPACK_SetPassed(const char *name)
{
  CLI::SetPassed(name);
}

/**
 * Reset the status of all timers.
 */
void MLPACK_ResetTimers()
{
  CLI::GetSingleton().timer.Reset();
}

/**
 * Enable timing.
 */
void MLPACK_EnableTimers()
{
  Timer::EnableTiming();
}

/**
 * Disable backtraces.
 */
void MLPACK_DisableBacktrace()
{
  Log::Fatal.backtrace = false;
}

/**
 * Turn verbose output on.
 */
void MLPACK_EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

/**
 * Turn verbose output off.
 */
void MLPACK_DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

/**
 * Clear settings.
 */
void MLPACK_ClearSettings()
{
  CLI::ClearSettings();
}

/**
 * Restore Settings.
 */
void MLPACK_RestoreSettings(const char *name)
{
  CLI::RestoreSettings(name);
}

} // extern C

} // namespace mlpack
