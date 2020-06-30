/**
 * @file bindings/julia/julia_util.cpp
 * @author Ryan Curtin
 *
 * Implementations of Julia binding functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/bindings/julia/julia_util.h>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <stdint.h>

using namespace mlpack;

extern "C" {

/**
 * Call CMD::RestoreSettings() for a given program name.
 */
void CMD_RestoreSettings(const char* programName)
{
  CMD::RestoreSettings(programName);
}

/**
 * Call CMD::SetParam<int>().
 */
void CMD_SetParamInt(const char* paramName, int paramValue)
{
  CMD::GetParam<int>(paramName) = paramValue;
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<double>().
 */
void CMD_SetParamDouble(const char* paramName, double paramValue)
{
  CMD::GetParam<double>(paramName) = paramValue;
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<std::string>().
 */
void CMD_SetParamString(const char* paramName, const char* paramValue)
{
  CMD::GetParam<std::string>(paramName) = paramValue;
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<bool>().
 */
void CMD_SetParamBool(const char* paramName, bool paramValue)
{
  CMD::GetParam<bool>(paramName) = paramValue;
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<std::vector<std::string>>() to set the length.
 */
void CMD_SetParamVectorStrLen(const char* paramName,
                              const size_t length)
{
  CMD::GetParam<std::vector<std::string>>(paramName).clear();
  CMD::GetParam<std::vector<std::string>>(paramName).resize(length);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<std::vector<std::string>>() to set an individual element.
 */
void CMD_SetParamVectorStrStr(const char* paramName,
                              const char* str,
                              const size_t element)
{
  CMD::GetParam<std::vector<std::string>>(paramName)[element] =
      std::string(str);
}

/**
 * Call CMD::SetParam<std::vector<int>>().
 */
void CMD_SetParamVectorInt(const char* paramName,
                           int* ints,
                           const size_t length)
{
  // Create a std::vector<int> object; unfortunately this requires copying the
  // vector elements.
  std::vector<int> vec;
  vec.resize(length);
  for (size_t i = 0; i < length; ++i)
    vec[i] = ints[i];

  CMD::GetParam<std::vector<int>>(paramName) = std::move(vec);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<arma::mat>().
 */
void CMD_SetParamMat(const char* paramName,
                     double* memptr,
                     const size_t rows,
                     const size_t cols,
                     const bool pointsAsRows)
{
  // Create the matrix as an alias.
  arma::mat m(memptr, arma::uword(rows), arma::uword(cols), false, true);
  CMD::GetParam<arma::mat>(paramName) = pointsAsRows ? m.t() : std::move(m);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<arma::Mat<size_t>>().
 */
void CMD_SetParamUMat(const char* paramName,
                      size_t* memptr,
                      const size_t rows,
                      const size_t cols,
                      const bool pointsAsRows)
{
  // Create the matrix as an alias.
  arma::Mat<size_t> m(memptr, arma::uword(rows), arma::uword(cols), false,
      true);
  CMD::GetParam<arma::Mat<size_t>>(paramName) = pointsAsRows ? m.t() :
      std::move(m);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<arma::rowvec>().
 */
void CMD_SetParamRow(const char* paramName,
                     double* memptr,
                     const size_t cols)
{
  arma::rowvec m(memptr, arma::uword(cols), false, true);
  CMD::GetParam<arma::rowvec>(paramName) = std::move(m);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<arma::Row<size_t>>().
 */
void CMD_SetParamURow(const char* paramName,
                      size_t* memptr,
                      const size_t cols)
{
  arma::Row<size_t> m(memptr, arma::uword(cols), false, true);
  CMD::GetParam<arma::Row<size_t>>(paramName) = std::move(m);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<arma::vec>().
 */
void CMD_SetParamCol(const char* paramName,
                     double* memptr,
                     const size_t rows)
{
  arma::vec m(memptr, arma::uword(rows), false, true);
  CMD::GetParam<arma::vec>(paramName) = std::move(m);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<arma::Row<size_t>>().
 */
void CMD_SetParamUCol(const char* paramName,
                      size_t* memptr,
                      const size_t rows)
{
  arma::Col<size_t> m(memptr, arma::uword(rows), false, true);
  CMD::GetParam<arma::Col<size_t>>(paramName) = std::move(m);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void CMD_SetParamMatWithInfo(const char* paramName,
                             bool* dimensions,
                             double* memptr,
                             const size_t rows,
                             const size_t cols,
                             const bool pointsAreRows)
{
  data::DatasetInfo d(pointsAreRows ? cols : rows);
  for (size_t i = 0; i < d.Dimensionality(); ++i)
  {
    d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
        data::Datatype::numeric;
  }

  arma::mat m(memptr, arma::uword(rows), arma::uword(cols), false, true);
  std::get<0>(CMD::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = std::move(d);
  std::get<1>(CMD::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = pointsAreRows ? std::move(m.t()) : std::move(m);
  CMD::SetPassed(paramName);
}

/**
 * Call CMD::GetParam<int>().
 */
int CMD_GetParamInt(const char* paramName)
{
  return CMD::GetParam<int>(paramName);
}

/**
 * Call CMD::GetParam<double>().
 */
double CMD_GetParamDouble(const char* paramName)
{
  return CMD::GetParam<double>(paramName);
}

/**
 * Call CMD::GetParam<std::string>().
 */
const char* CMD_GetParamString(const char* paramName)
{
  return CMD::GetParam<std::string>(paramName).c_str();
}

/**
 * Call CMD::GetParam<bool>().
 */
bool CMD_GetParamBool(const char* paramName)
{
  return CMD::GetParam<bool>(paramName);
}

/**
 * Call CMD::GetParam<std::vector<std::string>>() and get the length of the
 * vector.
 */
size_t CMD_GetParamVectorStrLen(const char* paramName)
{
  return CMD::GetParam<std::vector<std::string>>(paramName).size();
}

/**
 * Call CMD::GetParam<std::vector<std::string>>() and get the i'th string.
 */
const char* CMD_GetParamVectorStrStr(const char* paramName, const size_t i)
{
  return CMD::GetParam<std::vector<std::string>>(paramName)[i].c_str();
}

/**
 * Call CMD::GetParam<std::vector<int>>() and get the length of the vector.
 */
size_t CMD_GetParamVectorIntLen(const char* paramName)
{
  return CMD::GetParam<std::vector<int>>(paramName).size();
}

/**
 * Call CMD::GetParam<std::vector<int>>() and return a pointer to the vector.
 * The vector will be created in-place and it is expected that the calling
 * function will take ownership.
 */
int* CMD_GetParamVectorIntPtr(const char* paramName)
{
  const size_t size = CMD::GetParam<std::vector<int>>(paramName).size();
  int* ints = new int[size];

  for (size_t i = 0; i < size; ++i)
    ints[i] = CMD::GetParam<std::vector<int>>(paramName)[i];

  return ints;
}

/**
 * Get the number of rows in a matrix parameter.
 */
size_t CMD_GetParamMatRows(const char* paramName)
{
  return CMD::GetParam<arma::mat>(paramName).n_rows;
}

/**
 * Get the number of columns in a matrix parameter.
 */
size_t CMD_GetParamMatCols(const char* paramName)
{
  return CMD::GetParam<arma::mat>(paramName).n_cols;
}

/**
 * Get the memory pointer for a matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CMD_GetParamMat(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::mat& mat = CMD::GetParam<arma::mat>(paramName);
  if (mat.n_elem <= arma::arma_config::mat_prealloc)
  {
    // Copy the memory to something that we can give back to Julia.
    double* newMem = new double[mat.n_elem];
    arma::arrayops::copy(newMem, mat.mem, mat.n_elem);
    return newMem; // We believe Julia will free it.  Hopefully we are right.
  }
  else
  {
    arma::access::rw(mat.mem_state) = 1;
    return mat.memptr();
  }
}

/**
 * Get the number of rows in an unsigned matrix parameter.
 */
size_t CMD_GetParamUMatRows(const char* paramName)
{
  return CMD::GetParam<arma::Mat<size_t>>(paramName).n_rows;
}

/**
 * Get the number of columns in an unsigned matrix parameter.
 */
size_t CMD_GetParamUMatCols(const char* paramName)
{
  return CMD::GetParam<arma::Mat<size_t>>(paramName).n_cols;
}

/**
 * Get the memory pointer for an unsigned matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CMD_GetParamUMat(const char* paramName)
{
  arma::Mat<size_t>& mat = CMD::GetParam<arma::Mat<size_t>>(paramName);

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  if (mat.n_elem <= arma::arma_config::mat_prealloc)
  {
    // Copy the memory to something that we can give back to Julia.
    size_t* newMem = new size_t[mat.n_elem];
    arma::arrayops::copy(newMem, mat.mem, mat.n_elem);
    // We believe Julia will free it.  Hopefully we are right.
    return newMem;
  }
  else
  {
    arma::access::rw(mat.mem_state) = 1;
    return mat.memptr();
  }
}

/**
 * Get the number of rows in a column vector parameter.
 */
size_t CMD_GetParamColRows(const char* paramName)
{
  return CMD::GetParam<arma::vec>(paramName).n_rows;
}

/**
 * Get the memory pointer for a column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CMD_GetParamCol(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::vec& vec = CMD::GetParam<arma::vec>(paramName);
  if (vec.n_elem <= arma::arma_config::mat_prealloc)
  {
    // Copy the memory to something we can give back to Julia.
    double* newMem = new double[vec.n_elem];
    arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
    return newMem; // We believe Julia will free it.  Hopefully we are right.
  }
  else
  {
    arma::access::rw(vec.mem_state) = 1;
    return vec.memptr();
  }
}

/**
 * Get the number of columns in an unsigned column vector parameter.
 */
size_t CMD_GetParamUColRows(const char* paramName)
{
  return CMD::GetParam<arma::Col<size_t>>(paramName).n_rows;
}

/**
 * Get the memory pointer for an unsigned column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CMD_GetParamUCol(const char* paramName)
{
  arma::Col<size_t>& vec = CMD::GetParam<arma::Col<size_t>>(paramName);

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  if (vec.n_elem <= arma::arma_config::mat_prealloc)
  {
    // Copy the memory to something we can give back to Julia.
    size_t* newMem = new size_t[vec.n_elem];
    arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
    // We believe Julia will free it.  Hopefully we are right.
    return newMem;
  }
  else
  {
    arma::access::rw(vec.mem_state) = 1;
    return vec.memptr();
  }
}

/**
 * Get the number of columns in a row parameter.
 */
size_t CMD_GetParamRowCols(const char* paramName)
{
  return CMD::GetParam<arma::rowvec>(paramName).n_cols;
}

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CMD_GetParamRow(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::rowvec& vec = CMD::GetParam<arma::rowvec>(paramName);
  if (vec.n_elem <= arma::arma_config::mat_prealloc)
  {
    // Copy the memory to something we can give back to Julia.
    double* newMem = new double[vec.n_elem];
    arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
    return newMem;
  }
  else
  {
    arma::access::rw(vec.mem_state) = 1;
    return vec.memptr();
  }
}

/**
 * Get the number of columns in a row parameter.
 */
size_t CMD_GetParamURowCols(const char* paramName)
{
  return CMD::GetParam<arma::Row<size_t>>(paramName).n_cols;
}

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CMD_GetParamURow(const char* paramName)
{
  arma::Row<size_t>& vec = CMD::GetParam<arma::Row<size_t>>(paramName);

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  if (vec.n_elem <= arma::arma_config::mat_prealloc)
  {
    // Copy the memory to something we can give back to Julia.
    size_t* newMem = new size_t[vec.n_elem];
    arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
    return newMem;
  }
  else
  {
    arma::access::rw(vec.mem_state) = 1;
    return vec.memptr();
  }
}

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
size_t CMD_GetParamMatWithInfoRows(const char* paramName)
{
  return std::get<1>(CMD::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)).n_rows;
}

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
size_t CMD_GetParamMatWithInfoCols(const char* paramName)
{
  return std::get<1>(CMD::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)).n_cols;
}

/**
 * Get a pointer to an array of booleans representing whether or not dimensions
 * are categorical.  The calling function is expected to handle the memory
 * management.
 */
bool* CMD_GetParamMatWithInfoBoolPtr(const char* paramName)
{
  const data::DatasetInfo& d = std::get<0>(
      CMD::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName));

  bool* dims = new bool[d.Dimensionality()];
  for (size_t i = 0; i < d.Dimensionality(); ++i)
    dims[i] = (d.Type(i) == data::Datatype::numeric) ? false : true;

  return dims;
}

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
double* CMD_GetParamMatWithInfoPtr(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::mat& m = std::get<1>(
      CMD::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName));
  if (m.n_elem <= arma::arma_config::mat_prealloc)
  {
    double* newMem = new double[m.n_elem];
    arma::arrayops::copy(newMem, m.mem, m.n_elem);
    return newMem;
  }
  else
  {
    arma::access::rw(m.mem_state) = 1;
    return m.memptr();
  }
}

/**
 * Enable verbose output.
 */
void CMD_EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

/**
 * Disable verbose output.
 */
void CMD_DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

/**
 * Reset the state of all timers.
 */
void CMD_ResetTimers()
{
  CMD::GetSingleton().timer.Reset();
}

/**
 * Set an argument as passed to the CMD object.
 */
void CMD_SetPassed(const char* paramName)
{
  CMD::SetPassed(paramName);
}

} // extern "C"
