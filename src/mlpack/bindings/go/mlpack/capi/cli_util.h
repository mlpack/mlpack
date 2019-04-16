/**
 * @file cli_util.h
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
#ifndef MLPACK_BINDINGS_GO_MLPACK_CLI_UTIL_H
#define MLPACK_BINDINGS_GO_MLPACK_CLI_UTIL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * Set the double parameter to the given value.
 */
void mlpackSetParamDouble(const char* identifier, double value);

/**
 * Set the int parameter to the given value.
 */
void mlpackSetParamInt(const char* identifier, int value);

/**
 * Set the float parameter to the given value.
 */
void mlpackSetParamFloat(const char* identifier, float value);

/**
 * Set the bool parameter to the given value.
 */
void mlpackSetParamBool(const char* identifier, bool value);

/**
 * Set the string parameter to the given value.
 */
void mlpackSetParamString(const char* identifier, const char* value);

/**
 * Set the parameter to the given value, given that the type is a pointer.
 */
void mlpackSetParamPtr(const char* identifier,
                              const double* ptr);

/**
 * Set the int vector parameter to the given value.
 */
void mlpackSetParamVectorInt(const char* identifier,
                             const long long* ints,
                             const size_t length);

/**
 * Set the string vector parameter to the given value.
 */
void mlpackSetParamVectorStr(const char* identifier,
                             const char* str,
                             const size_t element);

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set the length.
 */
void mlpackSetParamVectorStrLen(const char* identifier,
                                const size_t length);

/**
 * Check if CLI has a specified parameter.
 */
bool mlpackHasParam(const char* identifier);

/**
 * Get the string parameter associated with specified identifier.
 */
const char* mlpackGetParamString(const char* identifier);

/**
 * Get the double parameter associated with specified identifier.
 */
double mlpackGetParamDouble(const char* identifier);

/**
 * Get the int parameter associated with specified identifier.
 */
int mlpackGetParamInt(const char* identifier);

/**
 * Get the bool parameter associated with specified identifier.
 */
bool mlpackGetParamBool(const char* identifier);

/**
 * Get the vector<int> parameter associated with specified identifier.
 */
void* mlpackGetVecIntPtr(const char* identifier);

/**
 * Get the vector<string> parameter associated with specified identifier.
 */
const char* mlpackGetVecStringPtr(const char* identifier, const size_t i);

/**
 * Get the vector<int> parameter's size.
 */
int mlpackVecIntSize(const char* identifier);

/**
 * Get the vector<string> parameter's size.
 */
int mlpackVecStringSize(const char* identifier);

/**
 * Set parameter as passed.
 */
void mlpackSetPassed(const char* name);

/**
 * Reset the status of all timers.
 */
void mlpackResetTimers();

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

/**
 * Clear settings.
 */
void mlpackClearSettings();

/**
 * Restore Settings.
 */
void mlpackRestoreSettings(const char* name);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
