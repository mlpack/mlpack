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
 * Get a new util::Params object, encoded as a heap-allocated void pointer.
 * You are responsible for freeing this!  You can use `DeleteParameters(void*)`.
 */
void* GetParameters(const char* bindingName);

/**
 * Delete a util::Params object that has been encoded as a void pointer.
 */
void DeleteParameters(void* p);

/**
 * Get a new util::Timers object, encoded as a heap-allocated void pointer.  You
 * are responsible for freeing this!  You can use `DeleteTimers(void*)`.
 */
void* Timers();

/**
 * Delete a util::Timers object that has been encoded as a void pointer.
 */
void DeleteTimers(void* t);

/**
 * Call params.SetParam<int>().
 */
void SetParamInt(void* params, const char* paramName, int paramValue);

/**
 * Call params.SetParam<double>().
 */
void SetParamDouble(void* params, const char* paramName, double paramValue);

/**
 * Call params.SetParam<std::string>().
 */
void SetParamString(void* params,
                    const char* paramName,
                    const char* paramValue);

/**
 * Call params.SetParam<bool>().
 */
void SetParamBool(void* params, const char* paramName, bool paramValue);

/**
 * Call params.SetParam<std::vector<std::string>>() to set the length.
 */
void SetParamVectorStrLen(void* params,
                          const char* paramName,
                          const size_t length);

/**
 * Call params.SetParam<std::vector<std::string>>() to set an individual
 * element.
 */
void SetParamVectorStrStr(void* params,
                          const char* paramName,
                          const char* str,
                          const size_t element);

/**
 * Call params.SetParam<std::vector<int>>().
 */
void SetParamVectorInt(void* params,
                       const char* paramName,
                       long long* ints,
                       const size_t length);

/**
 * Call params.SetParam<arma::mat>().
 */
void SetParamMat(void* params,
                 const char* paramName,
                 double* memptr,
                 const size_t rows,
                 const size_t cols,
                 const bool pointsAsRows);

/**
 * Call params.SetParam<arma::Mat<size_t>>().
 *
 * Note that we will have to allocate memory, since we must convert to a size_t
 * matrix, and we also subtract by one (since Julia uses 1-indexed labels).
 */
void SetParamUMat(void* params,
                  const char* paramName,
                  long long* memptr,
                  const size_t rows,
                  const size_t cols,
                  const bool pointsAsRows);

/**
 * Call params.SetParam<arma::rowvec>().
 */
void SetParamRow(void* params,
                 const char* paramName,
                 double* memptr,
                 const size_t cols);

/**
 * Call params.SetParam<arma::Row<size_t>>().
 *
 * Note that we will have to allocate memory, since we must convert to a size_t
 * vector, and we also subtract by one (since Julia uses 1-indexed labels).
 */
void SetParamURow(void* params,
                  const char* paramName,
                  long long* memptr,
                  const size_t cols);

/**
 * Call params.SetParam<arma::vec>().
 */
void SetParamCol(void* params,
                 const char* paramName,
                 double* memptr,
                 const size_t rows);

/**
 * Call params.SetParam<arma::Col<size_t>>().
 *
 * Note that we will have to allocate memory, since we must convert to a size_t
 * vector, and we also subtract by one (since Julia uses 1-indexed labels).
 */
void SetParamUCol(void* params,
                  const char* paramName,
                  long long* memptr,
                  const size_t rows);

/**
 * Call params.SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void SetParamMatWithInfo(void* params,
                         const char* paramName,
                         bool* dimensions,
                         double* memptr,
                         const size_t rows,
                         const size_t cols,
                         const bool pointsAreRows);

/**
 * Call params.GetParam<int>().
 */
int GetParamInt(void* params, const char* paramName);

/**
 * Call params.GetParam<double>().
 */
double GetParamDouble(void* params, const char* paramName);

/**
 * Call params.GetParam<std::string>().
 */
const char* GetParamString(void* params, const char* paramName);

/**
 * Call params.GetParam<bool>().
 */
bool GetParamBool(void* params, const char* paramName);

/**
 * Call params.GetParam<std::vector<std::string>>() and get the length of the
 * vector.
 */
size_t GetParamVectorStrLen(void* params, const char* paramName);

/**
 * Call params.GetParam<std::vector<std::string>>() and get the i'th string.
 */
const char* GetParamVectorStrStr(void* params,
                                 const char* paramName,
                                 const size_t i);

/**
 * Call params.GetParam<std::vector<int>>() and get the length of the vector.
 */
size_t GetParamVectorIntLen(void* params, const char* paramName);

/**
 * Call params.GetParam<std::vector<int>>() and return a pointer to the vector.
 * The vector will be created in-place and it is expected that the calling
 * function will take ownership.
 */
long long* GetParamVectorIntPtr(void* params, const char* paramName);

/**
 * Get the number of rows in a matrix parameter.
 */
size_t GetParamMatRows(void* params, const char* paramName);

/**
 * Get the number of columns in a matrix parameter.
 */
size_t GetParamMatCols(void* params, const char* paramName);

/**
 * Get the memory pointer for a matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* GetParamMat(void* params, const char* paramName);

/**
 * Get the number of rows in an unsigned matrix parameter.
 */
size_t GetParamUMatRows(void* params, const char* paramName);

/**
 * Get the number of columns in an unsigned matrix parameter.
 */
size_t GetParamUMatCols(void* params, const char* paramName);

/**
 * Get the memory pointer for an unsigned matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* GetParamUMat(void* params, const char* paramName);

/**
 * Get the number of rows in a column parameter.
 */
size_t GetParamColRows(void* params, const char* paramName);

/**
 * Get the memory pointer for a column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* GetParamCol(void* params, const char* paramName);

/**
 * Get the number of columns in an unsigned column vector parameter.
 */
size_t GetParamUColRows(void* params, const char* paramName);

/**
 * Get the memory pointer for an unsigned column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* GetParamUCol(void* params, const char* paramName);

/**
 * Get the number of columns in a row parameter.
 */
size_t GetParamRowCols(void* params, const char* paramName);

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* GetParamRow(void* params, const char* paramName);

/**
 * Get the number of columns in a row parameter.
 */
size_t GetParamURowCols(void* params, const char* paramName);

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* GetParamURow(void* params, const char* paramName);

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
size_t GetParamMatWithInfoRows(void* params, const char* paramName);

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
size_t GetParamMatWithInfoCols(void* params, const char* paramName);

/**
 * Get a pointer to an array of booleans representing whether or not dimensions
 * are categorical.  The calling function is expected to handle the memory
 * management.
 */
bool* GetParamMatWithInfoBoolPtr(void* params, const char* paramName);

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
double* GetParamMatWithInfoPtr(void* params, const char* paramName);

/**
 * Enable verbose output.
 */
void EnableVerbose();

/**
 * Disable verbose output.
 */
void DisableVerbose();

/**
 * Set an argument as passed to the IO object.
 */
void SetPassed(void* params, const char* paramName);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif
