/**
 * @file julia_util.h
 * @author Ryan Curtin
 *
 * Some utility functions in C that can be called from Julia with ccall() in
 * order to interact with the CLI interface.
 */
#ifndef MLPACK_BINDINGS_JULIA_JULIA_UTIL_H
#define MLPACK_BINDINGS_JULIA_JULIA_UTIL_H

#if defined(__cplusplus) || defined(c_plusplus)

#include <cstddef>
extern "C"
{
#else
#include <stddef.h>
#endif

/**
 * Call CLI::RestoreSettings() for a given program name.
 */
void CLI_RestoreSettings(const char* programName);

/**
 * Call CLI::SetParam<int>().
 */
void CLI_SetParamInt(const char* paramName, int paramValue);

/**
 * Call CLI::SetParam<double>().
 */
void CLI_SetParamDouble(const char* paramName, double paramValue);

/**
 * Call CLI::SetParam<std::string>().
 */
void CLI_SetParamString(const char* paramName, const char* paramValue);

/**
 * Call CLI::SetParam<bool>().
 */
void CLI_SetParamBool(const char* paramName, bool paramValue);

/**
 * Call CLI::SetParam<arma::mat>().
 */
void CLI_SetParamMat(const char* paramName,
                     double* memptr,
                     const size_t rows,
                     const size_t cols,
                     const bool pointsAsRows);

/**
 * Call CLI::SetParam<arma::Mat<size_t>>().
 */
void CLI_SetParamUmat(const char* paramName,
                      size_t* memptr,
                      const size_t rows,
                      const size_t cols,
                      const bool pointsAsRows);

/**
 * Get the number of rows in a matrix parameter.
 */
size_t CLI_GetParamMatRows(const char* paramName);

/**
 * Get the number of columns in a matrix parameter.
 */
size_t CLI_GetParamMatCols(const char* paramName);

/**
 * Get the memory pointer for a matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CLI_GetParamMat(const char* paramName);

/**
 * Enable verbose output.
 */
void CLI_EnableVerbose();

/**
 * Disable verbose output.
 */
void CLI_DisableVerbose();

/**
 * Reset timers.
 */
void CLI_ResetTimers();

/**
 * Set an argument as passed to the CLI object.
 */
void CLI_SetPassed(const char* paramName);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
