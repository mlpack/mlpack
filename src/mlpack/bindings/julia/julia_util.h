/**
 * @file julia/julia_util.h
 * @author Ryan Curtin
 *
 * Some utility functions in C that can be called from Julia with ccall() in
 * order to interact with the IO interface.
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
void IO_RestoreSettings(const char* programName);

/**
 * Call CLI::SetParam<int>().
 */
void IO_SetParamInt(const char* paramName, int paramValue);

/**
 * Call CLI::SetParam<double>().
 */
void IO_SetParamDouble(const char* paramName, double paramValue);

/**
 * Call CLI::SetParam<std::string>().
 */
void IO_SetParamString(const char* paramName, const char* paramValue);

/**
 * Call CLI::SetParam<bool>().
 */
void IO_SetParamBool(const char* paramName, bool paramValue);

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set the length.
 */
void IO_SetParamVectorStrLen(const char* paramName,
                              const size_t length);

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set an individual element.
 */
void IO_SetParamVectorStrStr(const char* paramName,
                              const char* str,
                              const size_t element);

/**
 * Call CLI::SetParam<std::vector<int>>().
 */
void IO_SetParamVectorInt(const char* paramName,
                           int* ints,
                           const size_t length);

/**
 * Call CLI::SetParam<arma::mat>().
 */
void IO_SetParamMat(const char* paramName,
                     double* memptr,
                     const size_t rows,
                     const size_t cols,
                     const bool pointsAsRows);

/**
 * Call CLI::SetParam<arma::Mat<size_t>>().
 */
void IO_SetParamUMat(const char* paramName,
                      size_t* memptr,
                      const size_t rows,
                      const size_t cols,
                      const bool pointsAsRows);

/**
 * Call CLI::SetParam<arma::rowvec>().
 */
void IO_SetParamRow(const char* paramName,
                     double* memptr,
                     const size_t cols);

/**
 * Call CLI::SetParam<arma::Row<size_t>>().
 */
void IO_SetParamURow(const char* paramName,
                      size_t* memptr,
                      const size_t cols);

/**
 * Call CLI::SetParam<arma::vec>().
 */
void IO_SetParamCol(const char* paramName,
                     double* memptr,
                     const size_t rows);

/**
 * Call CLI::SetParam<arma::Col<size_t>>().
 */
void IO_SetParamUCol(const char* paramName,
                      size_t* memptr,
                      const size_t rows);

/**
 * Call CLI::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void IO_SetParamMatWithInfo(const char* paramName,
                             bool* dimensions,
                             double* memptr,
                             const size_t rows,
                             const size_t cols,
                             const bool pointsAreRows);

/**
 * Call CLI::GetParam<int>().
 */
int IO_GetParamInt(const char* paramName);

/**
 * Call CLI::GetParam<double>().
 */
double IO_GetParamDouble(const char* paramName);

/**
 * Call CLI::GetParam<std::string>().
 */
const char* IO_GetParamString(const char* paramName);

/**
 * Call CLI::GetParam<bool>().
 */
bool IO_GetParamBool(const char* paramName);

/**
 * Call CLI::GetParam<std::vector<std::string>>() and get the length of the
 * vector.
 */
size_t IO_GetParamVectorStrLen(const char* paramName);

/**
 * Call CLI::GetParam<std::vector<std::string>>() and get the i'th string.
 */
const char* IO_GetParamVectorStrStr(const char* paramName, const size_t i);

/**
 * Call CLI::GetParam<std::vector<int>>() and get the length of the vector.
 */
size_t IO_GetParamVectorIntLen(const char* paramName);

/**
 * Call CLI::GetParam<std::vector<int>>() and return a pointer to the vector.
 * The vector will be created in-place and it is expected that the calling
 * function will take ownership.
 */
int* IO_GetParamVectorIntPtr(const char* paramName);

/**
 * Get the number of rows in a matrix parameter.
 */
size_t IO_GetParamMatRows(const char* paramName);

/**
 * Get the number of columns in a matrix parameter.
 */
size_t IO_GetParamMatCols(const char* paramName);

/**
 * Get the memory pointer for a matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* IO_GetParamMat(const char* paramName);

/**
 * Get the number of rows in an unsigned matrix parameter.
 */
size_t IO_GetParamUMatRows(const char* paramName);

/**
 * Get the number of columns in an unsigned matrix parameter.
 */
size_t IO_GetParamUMatCols(const char* paramName);

/**
 * Get the memory pointer for an unsigned matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* IO_GetParamUMat(const char* paramName);

/**
 * Get the number of rows in a column parameter.
 */
size_t IO_GetParamColRows(const char* paramName);

/**
 * Get the memory pointer for a column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* IO_GetParamCol(const char* paramName);

/**
 * Get the number of columns in an unsigned column vector parameter.
 */
size_t IO_GetParamUColRows(const char* paramName);

/**
 * Get the memory pointer for an unsigned column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* IO_GetParamUCol(const char* paramName);

/**
 * Get the number of columns in a row parameter.
 */
size_t IO_GetParamRowCols(const char* paramName);

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* IO_GetParamRow(const char* paramName);

/**
 * Get the number of columns in a row parameter.
 */
size_t IO_GetParamURowCols(const char* paramName);

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* IO_GetParamURow(const char* paramName);

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
size_t IO_GetParamMatWithInfoRows(const char* paramName);

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
size_t IO_GetParamMatWithInfoCols(const char* paramName);

/**
 * Get a pointer to an array of booleans representing whether or not dimensions
 * are categorical.  The calling function is expected to handle the memory
 * management.
 */
bool* IO_GetParamMatWithInfoBoolPtr(const char* paramName);

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
double* IO_GetParamMatWithInfoPtr(const char* paramName);

/**
 * Enable verbose output.
 */
void IO_EnableVerbose();

/**
 * Disable verbose output.
 */
void IO_DisableVerbose();

/**
 * Reset timers.
 */
void IO_ResetTimers();

/**
 * Set an argument as passed to the IO object.
 */
void IO_SetPassed(const char* paramName);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
