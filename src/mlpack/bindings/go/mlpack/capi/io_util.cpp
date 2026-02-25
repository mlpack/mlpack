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
 * Get a new Params object for the given binding name.
 */
void* mlpackGetParams(const char* bindingName)
{
  util::Params* p = new util::Params(IO::Parameters(bindingName));
  return (void*) p;
}

/**
 * Get a new Timers object.
 */
void* mlpackGetTimers()
{
  util::Timers* t = new util::Timers();
  return (void*) t;
}

/**
 * Delete the given Params object.
 */
void mlpackCleanParams(void* params)
{
  util::Params* p = (util::Params*) params;
  delete p;
}

/**
 * Delete the given Timers object.
 */
void mlpackCleanTimers(void* timers)
{
  util::Timers* t = (util::Timers*) timers;
  delete t;
}

/**
 * Set the double parameter to the given value.
 */
void mlpackSetParamDouble(void* params, const char* identifier, double value)
{
  util::Params& p = *((util::Params*) params);
  util::SetParam(p, identifier, value);
}

/**
 * Set the int parameter to the given value.
 */
void mlpackSetParamInt(void* params, const char* identifier, int value)
{
  util::Params& p = *((util::Params*) params);
  util::SetParam(p, identifier, value);
}

/**
 * Set the float parameter to the given value.
 */
void mlpackSetParamFloat(void* params, const char* identifier, float value)
{
  util::Params& p = *((util::Params*) params);
  util::SetParam(p, identifier, value);
}

/**
 * Set the bool parameter to the given value.
 */
void mlpackSetParamBool(void* params, const char* identifier, bool value)
{
  util::Params& p = *((util::Params*) params);
  util::SetParam(p, identifier, value);
}

/**
 * Set the string parameter to the given value.
 */
void mlpackSetParamString(void* params,
                          const char* identifier,
                          const char* value)
{
  util::Params& p = *((util::Params*) params);
  p.Get<std::string>(identifier) = value;
}

/**
 * Set the int vector parameter to the given value.
 */
void mlpackSetParamVectorInt(void* params,
                             const char* identifier,
                             const long long* ints,
                             const size_t length)
{
  util::Params& p = *((util::Params*) params);

  // Create a std::vector<int> object; unfortunately this requires copying the
  // vector elements.
  std::vector<int> vec(length);
  for (size_t i = 0; i < length; ++i)
    vec[i] = ints[i];

  p.Get<std::vector<int>>(identifier) = std::move(vec);
  p.SetPassed(identifier);
}

/**
 * Call IO::SetParam<std::vector<std::string>>() to set the length.
 */
void mlpackSetParamVectorStrLen(void* params,
                                const char* identifier,
                                const size_t length)
{
  util::Params& p = *((util::Params*) params);
  p.Get<std::vector<std::string>>(identifier).clear();
  p.Get<std::vector<std::string>>(identifier).resize(length);
  p.SetPassed(identifier);
}

/**
 * Set the string vector parameter to the given value.
 */
void mlpackSetParamVectorStr(void* params,
                             const char* identifier,
                             const char* str,
                             const size_t element)
{
  util::Params& p = *((util::Params*) params);
  p.Get<std::vector<std::string>>(identifier)[element] = std::string(str);
}

/**
 * Set the parameter to the given value, given that the type is a pointer.
 */
void mlpackSetParamPtr(void* params,
                       const char* identifier,
                       double* ptr)
{
  util::Params& p = *((util::Params*) params);
  util::SetParamPtr(p, identifier, ptr);
}

/**
 * Check if IO has a specified parameter.
 */
bool mlpackHasParam(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Has(identifier);
}

/**
 * Get the string parameter associated with specified identifier.
 */
const char* mlpackGetParamString(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<std::string>(identifier).c_str();
}

/**
 * Get the double parameter associated with specified identifier.
 */
double mlpackGetParamDouble(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<double>(identifier);
}

/**
 * Get the int parameter associated with specified identifier.
 */
int mlpackGetParamInt(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<int>(identifier);
}

/**
 * Get the bool parameter associated with specified identifier.
 */
bool mlpackGetParamBool(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<bool>(identifier);
}

/**
 * Get the vector<int> parameter associated with specified identifier.
 */
void* mlpackGetVecIntPtr(void* params, const char* identifier)
{
  const size_t size = mlpackVecIntSize(params, identifier);
  long long* ints = new long long[size];

  util::Params& p = *((util::Params*) params);
  for (size_t i = 0; i < size; i++)
    ints[i] = p.Get<std::vector<int>>(identifier)[i];

  return ints;
}

/**
 * Get the vector<string> parameter associated with specified identifier.
 */
const char* mlpackGetVecStringPtr(void* params,
                                  const char* identifier,
                                  const size_t i)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<std::vector<std::string>>(identifier)[i].c_str();
}

/**
 * Get the vector<int> parameter's size.
 */
int mlpackVecIntSize(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<std::vector<int>>(identifier).size();
}

/**
 * Get the vector<string> parameter's size.
 */
int mlpackVecStringSize(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<std::vector<std::string>>(identifier).size();
}

/**
 * Set parameter as passed.
 */
void mlpackSetPassed(void* params, const char* name)
{
  util::Params& p = *((util::Params*) params);
  p.SetPassed(name);
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

} // extern C

} // namespace mlpack
