/**
 * @file bindings/go/mlpack/capi/io_util.h
 * @author Yasmine Dumouchel
 * @author Yashwant Singh
 *
 * Header file for cgo to call C functions from go.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_MLPACK_IO_UTIL_H
#define MLPACK_BINDINGS_GO_MLPACK_IO_UTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Get a new Params object for the given binding name.
 */
void* mlpackGetParams(const char* bindingName);

/**
 * Get a new Timers object.
 */
void* mlpackGetTimers();

/**
 * Delete the given Params object.
 */
void mlpackCleanParams(void* params);

/**
 * Delete the given Timers object.
 */
void mlpackCleanTimers(void* timers);

/**
 * Set the double parameter to the given value.
 */
void mlpackSetParamDouble(void* params, const char* identifier, double value);

/**
 * Set the int parameter to the given value.
 */
void mlpackSetParamInt(void* params, const char* identifier, int value);

/**
 * Set the float parameter to the given value.
 */
void mlpackSetParamFloat(void* params, const char* identifier, float value);

/**
 * Set the bool parameter to the given value.
 */
void mlpackSetParamBool(void* params, const char* identifier, bool value);

/**
 * Set the string parameter to the given value.
 */
void mlpackSetParamString(void* params,
                          const char* identifier,
                          const char* value);

/**
 * Set the parameter to the given value, given that the type is a pointer.
 */
void mlpackSetParamPtr(void* params, const char* identifier, double* ptr);

/**
 * Set the int vector parameter to the given value.
 */
void mlpackSetParamVectorInt(void* params,
                             const char* identifier,
                             const long long* ints,
                             const size_t length);

/**
 * Set the string vector parameter to the given value.
 */
void mlpackSetParamVectorStr(void* params,
                             const char* identifier,
                             const char* str,
                             const size_t element);

/**
 * Call IO::SetParam<std::vector<std::string>>() to set the length.
 */
void mlpackSetParamVectorStrLen(void* params,
                                const char* identifier,
                                const size_t length);

/**
 * Check if IO has a specified parameter.
 */
bool mlpackHasParam(void* params, const char* identifier);

/**
 * Get the string parameter associated with specified identifier.
 */
const char* mlpackGetParamString(void* params, const char* identifier);

/**
 * Get the double parameter associated with specified identifier.
 */
double mlpackGetParamDouble(void* params, const char* identifier);

/**
 * Get the int parameter associated with specified identifier.
 */
int mlpackGetParamInt(void* params, const char* identifier);

/**
 * Get the bool parameter associated with specified identifier.
 */
bool mlpackGetParamBool(void* params, const char* identifier);

/**
 * Get the vector<int> parameter associated with specified identifier.
 */
void* mlpackGetVecIntPtr(void* params, const char* identifier);

/**
 * Get the vector<string> parameter associated with specified identifier.
 */
const char* mlpackGetVecStringPtr(void* params,
                                  const char* identifier,
                                  const size_t i);

/**
 * Get the vector<int> parameter's size.
 */
int mlpackVecIntSize(void* params, const char* identifier);

/**
 * Get the vector<string> parameter's size.
 */
int mlpackVecStringSize(void* params, const char* identifier);

/**
 * Set parameter as passed.
 */
void mlpackSetPassed(void* params, const char* name);

/**
 * Enable timing.
 */
void mlpackEnableTimers();

/**
 * Disable backtraces.
 */
void mlpackDisableBacktrace();

/**
 * Turn verbose output on.
 */
void mlpackEnableVerbose();

/**
 * Turn verbose output off.
 */
void mlpackDisableVerbose();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
