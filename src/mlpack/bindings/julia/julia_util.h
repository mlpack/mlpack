/**
 * @file julia/julia_util.h
 * @author Ryan Curtin
 *
 * Some utility functions in C that can be called from Julia with ccall() in
 * order to interact with the CMD interface.
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
 * Call CMD::RestoreSettings() for a given program name.
 */
void CMD_RestoreSettings(const char* programName);

/**
 * Call CMD::SetParam<int>().
 */
void CMD_SetParamInt(const char* paramName, int paramValue);

/**
 * Call CMD::SetParam<double>().
 */
void CMD_SetParamDouble(const char* paramName, double paramValue);

/**
 * Call CMD::SetParam<std::string>().
 */
void CMD_SetParamString(const char* paramName, const char* paramValue);

/**
 * Call CMD::SetParam<bool>().
 */
void CMD_SetParamBool(const char* paramName, bool paramValue);

/**
 * Call CMD::SetParam<std::vector<std::string>>() to set the length.
 */
void CMD_SetParamVectorStrLen(const char* paramName,
                              const size_t length);

/**
 * Call CMD::SetParam<std::vector<std::string>>() to set an individual element.
 */
void CMD_SetParamVectorStrStr(const char* paramName,
                              const char* str,
                              const size_t element);

/**
 * Call CMD::SetParam<std::vector<int>>().
 */
void CMD_SetParamVectorInt(const char* paramName,
                           int* ints,
                           const size_t length);

/**
 * Call CMD::SetParam<arma::mat>().
 */
void CMD_SetParamMat(const char* paramName,
                     double* memptr,
                     const size_t rows,
                     const size_t cols,
                     const bool pointsAsRows);

/**
 * Call CMD::SetParam<arma::Mat<size_t>>().
 */
void CMD_SetParamUMat(const char* paramName,
                      size_t* memptr,
                      const size_t rows,
                      const size_t cols,
                      const bool pointsAsRows);

/**
 * Call CMD::SetParam<arma::rowvec>().
 */
void CMD_SetParamRow(const char* paramName,
                     double* memptr,
                     const size_t cols);

/**
 * Call CMD::SetParam<arma::Row<size_t>>().
 */
void CMD_SetParamURow(const char* paramName,
                      size_t* memptr,
                      const size_t cols);

/**
 * Call CMD::SetParam<arma::vec>().
 */
void CMD_SetParamCol(const char* paramName,
                     double* memptr,
                     const size_t rows);

/**
 * Call CMD::SetParam<arma::Col<size_t>>().
 */
void CMD_SetParamUCol(const char* paramName,
                      size_t* memptr,
                      const size_t rows);

/**
 * Call CMD::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void CMD_SetParamMatWithInfo(const char* paramName,
                             bool* dimensions,
                             double* memptr,
                             const size_t rows,
                             const size_t cols,
                             const bool pointsAreRows);

/**
 * Call CMD::GetParam<int>().
 */
int CMD_GetParamInt(const char* paramName);

/**
 * Call CMD::GetParam<double>().
 */
double CMD_GetParamDouble(const char* paramName);

/**
 * Call CMD::GetParam<std::string>().
 */
const char* CMD_GetParamString(const char* paramName);

/**
 * Call CMD::GetParam<bool>().
 */
bool CMD_GetParamBool(const char* paramName);

/**
 * Call CMD::GetParam<std::vector<std::string>>() and get the length of the
 * vector.
 */
size_t CMD_GetParamVectorStrLen(const char* paramName);

/**
 * Call CMD::GetParam<std::vector<std::string>>() and get the i'th string.
 */
const char* CMD_GetParamVectorStrStr(const char* paramName, const size_t i);

/**
 * Call CMD::GetParam<std::vector<int>>() and get the length of the vector.
 */
size_t CMD_GetParamVectorIntLen(const char* paramName);

/**
 * Call CMD::GetParam<std::vector<int>>() and return a pointer to the vector.
 * The vector will be created in-place and it is expected that the calling
 * function will take ownership.
 */
int* CMD_GetParamVectorIntPtr(const char* paramName);

/**
 * Get the number of rows in a matrix parameter.
 */
size_t CMD_GetParamMatRows(const char* paramName);

/**
 * Get the number of columns in a matrix parameter.
 */
size_t CMD_GetParamMatCols(const char* paramName);

/**
 * Get the memory pointer for a matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CMD_GetParamMat(const char* paramName);

/**
 * Get the number of rows in an unsigned matrix parameter.
 */
size_t CMD_GetParamUMatRows(const char* paramName);

/**
 * Get the number of columns in an unsigned matrix parameter.
 */
size_t CMD_GetParamUMatCols(const char* paramName);

/**
 * Get the memory pointer for an unsigned matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CMD_GetParamUMat(const char* paramName);

/**
 * Get the number of rows in a column parameter.
 */
size_t CMD_GetParamColRows(const char* paramName);

/**
 * Get the memory pointer for a column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CMD_GetParamCol(const char* paramName);

/**
 * Get the number of columns in an unsigned column vector parameter.
 */
size_t CMD_GetParamUColRows(const char* paramName);

/**
 * Get the memory pointer for an unsigned column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CMD_GetParamUCol(const char* paramName);

/**
 * Get the number of columns in a row parameter.
 */
size_t CMD_GetParamRowCols(const char* paramName);

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CMD_GetParamRow(const char* paramName);

/**
 * Get the number of columns in a row parameter.
 */
size_t CMD_GetParamURowCols(const char* paramName);

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CMD_GetParamURow(const char* paramName);

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
size_t CMD_GetParamMatWithInfoRows(const char* paramName);

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
size_t CMD_GetParamMatWithInfoCols(const char* paramName);

/**
 * Get a pointer to an array of booleans representing whether or not dimensions
 * are categorical.  The calling function is expected to handle the memory
 * management.
 */
bool* CMD_GetParamMatWithInfoBoolPtr(const char* paramName);

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
double* CMD_GetParamMatWithInfoPtr(const char* paramName);

/**
 * Enable verbose output.
 */
void CMD_EnableVerbose();

/**
 * Disable verbose output.
 */
void CMD_DisableVerbose();

/**
 * Reset timers.
 */
void CMD_ResetTimers();

/**
 * Set an argument as passed to the CMD object.
 */
void CMD_SetPassed(const char* paramName);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
