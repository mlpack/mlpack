/**
 * @file cli_util.h
 * @author Yasmine Dumouchel
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
extern void MLPACK_SetParamDouble(const char *identifier, double value);

/**
 * Set the int parameter to the given value.
 */
extern void MLPACK_SetParamInt(const char *identifier, int value);

/**
 * Set the float parameter to the given value.
 */
extern void MLPACK_SetParamFloat(const char *identifier, float value);

/**
 * Set the bool parameter to the given value.
 */
extern void MLPACK_SetParamBool(const char *identifier, bool value);

/**
 * Set the string parameter to the given value.
 */
extern void MLPACK_SetParamString(const char *identifier, const char *value);

/**
 * Set the parameter to the given value, given that the type is a pointer.
 */
extern void MLPACK_SetParamPtr(const char *identifier, const double *ptr, const bool copy);

/**
 * Check if CLI has a specified parameter.
 */
extern bool MLPACK_HasParam(const char *identifier);

/**
 * Get the string parameter associated with specified identifier.
 */
extern char *MLPACK_GetParamString(const char *identifier);

/**
 * Get the double parameter associated with specified identifier.
 */
extern double MLPACK_GetParamDouble(const char *identifier);

/**
 * Get the int parameter associated with specified identifier.
 */
extern int MLPACK_GetParamInt(const char *identifier);

/**
 * Get the bool parameter associated with specified identifier.
 */
extern bool MLPACK_GetParamBool(const char *identifier);

/**
 * Get the vector<int> parameter associated with specified identifier.
 */
extern void *MLPACK_GetVecIntPtr(const char *identifier);

/**
 * Get the vector<string> parameter associated with specified identifier.
 */
extern void *MLPACK_GetVecStringPtr(const char *identifier);

/**
 * Get the vector<int> parameter's size.
 */
extern int MLPACK_VecIntSize(const char *identifier);

/**
 * Get the vector<string> parameter's size.
 */
extern int MLPACK_VecStringSize(const char *identifier);

/**
 * Set parameter as passed.
 */
extern void MLPACK_SetPassed(const char *name);

/**
 * Reset the status of all timers.
 */
extern void MLPACK_ResetTimers();

/**
 * Enable timing.
 */
extern void MLPACK_EnableTimers();

/**
 * Disable backtraces.
 */
extern void MLPACK_DisableBacktrace();

/**
 * Turn verbose output on.
 */
extern void MLPACK_EnableVerbose();

/**
 * Turn verbose output off.
 */
extern void MLPACK_DisableVerbose();

/**
 * Clear settings.
 */
extern void MLPACK_ClearSettings();

/**
 * Restore Settings.
 */
extern void MLPACK_RestoreSettings(const char *name);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
