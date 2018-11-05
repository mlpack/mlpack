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
#include <cstdint>
extern "C"
{
#else
#include <stddef.h>
#include <stdint.h>
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
 * Call CLI::SetParam<std::vector<std::string>>() to set the length.
 */
void CLI_SetParamVectorStrLen(const char* paramName,
                              const size_t length);

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set an individual element.
 */
void CLI_SetParamVectorStrStr(const char* paramName,
                              const char* str,
                              const size_t element);

/**
 * Call CLI::SetParam<std::vector<int>>().
 */
void CLI_SetParamVectorInt(const char* paramName,
                           uint64_t* ints,
                           const size_t length);

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
void CLI_SetParamUMat(const char* paramName,
                      size_t* memptr,
                      const size_t rows,
                      const size_t cols,
                      const bool pointsAsRows);

/**
 * Call CLI::SetParam<arma::rowvec>().
 */
void CLI_SetParamRow(const char* paramName,
                     double* memptr,
                     const size_t cols);

/**
 * Call CLI::SetParam<arma::Row<size_t>>().
 */
void CLI_SetParamURow(const char* paramName,
                      size_t* memptr,
                      const size_t cols);

/**
 * Call CLI::SetParam<arma::vec>().
 */
void CLI_SetParamCol(const char* paramName,
                     double* memptr,
                     const size_t rows);

/**
 * Call CLI::SetParam<arma::Col<size_t>>().
 */
void CLI_SetParamUCol(const char* paramName,
                      size_t* memptr,
                      const size_t rows);

/**
 * Call CLI::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void CLI_SetParamMatWithInfo(const char* paramName,
                             bool* dimensions,
                             double* memptr,
                             const size_t rows,
                             const size_t cols,
                             const bool pointsAreRows);

/**
 * Call CLI::GetParam<int>().
 */
int CLI_GetParamInt(const char* paramName);

/**
 * Call CLI::GetParam<double>().
 */
double CLI_GetParamDouble(const char* paramName);

/**
 * Call CLI::GetParam<std::string>().
 */
const char* CLI_GetParamString(const char* paramName);

/**
 * Call CLI::GetParam<bool>().
 */
bool CLI_GetParamBool(const char* paramName);

/**
 * Call CLI::GetParam<std::vector<std::string>>() and get the length of the
 * vector.
 */
size_t CLI_GetParamVectorStrLen(const char* paramName);

/**
 * Call CLI::GetParam<std::vector<std::string>>() and get the i'th string.
 */
const char* CLI_GetParamVectorStrStr(const char* paramName, const int i);

/**
 * Call CLI::GetParam<std::vector<int>>() and get the length of the vector.
 */
size_t CLI_GetParamVectorIntLen(const char* paramName);

/**
 * Call CLI::GetParam<std::vector<int>>() and return a pointer to the vector.
 * The vector will be created in-place and it is expected that the calling
 * function will take ownership.
 */
uint64_t* CLI_GetParamVectorIntPtr(const char* paramName);

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
 * Get the number of rows in an unsigned matrix parameter.
 */
size_t CLI_GetParamUMatRows(const char* paramName);

/**
 * Get the number of columns in an unsigned matrix parameter.
 */
size_t CLI_GetParamUMatCols(const char* paramName);

/**
 * Get the memory pointer for an unsigned matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CLI_GetParamUMat(const char* paramName);

/**
 * Get the number of rows in a column parameter.
 */
size_t CLI_GetParamColRows(const char* paramName);

/**
 * Get the memory pointer for a column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CLI_GetParamCol(const char* paramName);

/**
 * Get the number of columns in an unsigned column vector parameter.
 */
size_t CLI_GetParamUColRows(const char* paramName);

/**
 * Get the memory pointer for an unsigned column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CLI_GetParamUCol(const char* paramName);

/**
 * Get the number of columns in a row parameter.
 */
size_t CLI_GetParamRowCols(const char* paramName);

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CLI_GetParamRow(const char* paramName);

/**
 * Get the number of columns in a row parameter.
 */
size_t CLI_GetParamURowCols(const char* paramName);

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CLI_GetParamURow(const char* paramName);

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
size_t CLI_GetParamMatWithInfoRows(const char* paramName);

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
size_t CLI_GetParamMatWithInfoCols(const char* paramName);

/**
 * Get a pointer to an array of booleans representing whether or not dimensions
 * are categorical.  The calling function is expected to handle the memory
 * management.
 */
bool* CLI_GetParamMatWithInfoBoolPtr(const char* paramName);

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
double* CLI_GetParamMatWithInfoPtr(const char* paramName);

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
