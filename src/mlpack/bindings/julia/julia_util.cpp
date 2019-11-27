/**
 * @file julia_util.cpp
 * @author Ryan Curtin
 *
 * Implementations of Julia binding functionality.
 */
#include <mlpack/bindings/julia/julia_util.h>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>

using namespace mlpack;

extern "C" {

/**
 * Call CLI::RestoreSettings() for a given program name.
 */
void CLI_RestoreSettings(const char* programName)
{
  CLI::RestoreSettings(programName);
}

/**
 * Call CLI::SetParam<int>().
 */
void CLI_SetParamInt(const char* paramName, int paramValue)
{
  CLI::GetParam<int>(paramName) = paramValue;
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<double>().
 */
void CLI_SetParamDouble(const char* paramName, double paramValue)
{
  CLI::GetParam<double>(paramName) = paramValue;
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<std::string>().
 */
void CLI_SetParamString(const char* paramName, const char* paramValue)
{
  CLI::GetParam<std::string>(paramName) = paramValue;
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<bool>().
 */
void CLI_SetParamBool(const char* paramName, bool paramValue)
{
  CLI::GetParam<bool>(paramName) = paramValue;
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set the length.
 */
void CLI_SetParamVectorStrLen(const char* paramName,
                              const size_t length)
{
  CLI::GetParam<std::vector<std::string>>(paramName).clear();
  CLI::GetParam<std::vector<std::string>>(paramName).resize(length);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set an individual element.
 */
void CLI_SetParamVectorStrStr(const char* paramName,
                              const char* str,
                              const size_t element)
{
  CLI::GetParam<std::vector<std::string>>(paramName)[element] =
      std::string(str);
}

/**
 * Call CLI::SetParam<std::vector<int>>().
 */
void CLI_SetParamVectorInt(const char* paramName,
                           uint64_t* ints,
                           const size_t length)
{
  // Create a std::vector<int> object; unfortunately this requires copying the
  // vector elements.
  std::vector<int> vec(length);
  for (size_t i = 0; i < (size_t) length; ++i)
    vec[i] = ints[i];

  CLI::GetParam<std::vector<int>>(paramName) = std::move(vec);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::mat>().
 */
void CLI_SetParamMat(const char* paramName,
                     double* memptr,
                     const size_t rows,
                     const size_t cols,
                     const bool pointsAsRows)
{
  // Create the matrix as an alias.
  arma::mat m(memptr, rows, cols, false, true);
  CLI::GetParam<arma::mat>(paramName) = pointsAsRows ? m.t() : std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::Mat<size_t>>().
 */
void CLI_SetParamUMat(const char* paramName,
                      size_t* memptr,
                      const size_t rows,
                      const size_t cols,
                      const bool pointsAsRows)
{
  // Create the matrix as an alias.
  arma::Mat<size_t> m(memptr, rows, cols, false, true);
  CLI::GetParam<arma::Mat<size_t>>(paramName) = pointsAsRows ? m.t() :
      std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::rowvec>().
 */
void CLI_SetParamRow(const char* paramName,
                     double* memptr,
                     const size_t cols)
{
  arma::rowvec m(memptr, cols, false, true);
  CLI::GetParam<arma::rowvec>(paramName) = std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::Row<size_t>>().
 */
void CLI_SetParamURow(const char* paramName,
                      size_t* memptr,
                      const size_t cols)
{
  arma::Row<size_t> m(memptr, cols, false, true);
  CLI::GetParam<arma::Row<size_t>>(paramName) = std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::vec>().
 */
void CLI_SetParamCol(const char* paramName,
                     double* memptr,
                     const size_t rows)
{
  arma::vec m(memptr, rows, false, true);
  CLI::GetParam<arma::vec>(paramName) = std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::Row<size_t>>().
 */
void CLI_SetParamUCol(const char* paramName,
                     size_t* memptr,
                     const size_t rows)
{
  arma::Col<size_t> m(memptr, rows, false, true);
  CLI::GetParam<arma::Col<size_t>>(paramName) = std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void CLI_SetParamMatWithInfo(const char* paramName,
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

  arma::mat m(memptr, rows, cols, false, true);
  std::get<0>(CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = std::move(d);
  std::get<1>(CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = pointsAreRows ? std::move(m.t()) : std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::GetParam<int>().
 */
int CLI_GetParamInt(const char* paramName)
{
  return CLI::GetParam<int>(paramName);
}

/**
 * Call CLI::GetParam<double>().
 */
double CLI_GetParamDouble(const char* paramName)
{
  return CLI::GetParam<double>(paramName);
}

/**
 * Call CLI::GetParam<std::string>().
 */
const char* CLI_GetParamString(const char* paramName)
{
  return CLI::GetParam<std::string>(paramName).c_str();
}

/**
 * Call CLI::GetParam<bool>().
 */
bool CLI_GetParamBool(const char* paramName)
{
  return CLI::GetParam<bool>(paramName);
}

/**
 * Call CLI::GetParam<std::vector<std::string>>() and get the length of the
 * vector.
 */
size_t CLI_GetParamVectorStrLen(const char* paramName)
{
  return CLI::GetParam<std::vector<std::string>>(paramName).size();
}

/**
 * Call CLI::GetParam<std::vector<std::string>>() and get the i'th string.
 */
const char* CLI_GetParamVectorStrStr(const char* paramName, const int i)
{
  return CLI::GetParam<std::vector<std::string>>(paramName)[i].c_str();
}

/**
 * Call CLI::GetParam<std::vector<int>>() and get the length of the vector.
 */
size_t CLI_GetParamVectorIntLen(const char* paramName)
{
  return CLI::GetParam<std::vector<int>>(paramName).size();
}

/**
 * Call CLI::GetParam<std::vector<int>>() and return a pointer to the vector.
 * The vector will be created in-place and it is expected that the calling
 * function will take ownership.
 */
uint64_t* CLI_GetParamVectorIntPtr(const char* paramName)
{
  const size_t size = CLI::GetParam<std::vector<int>>(paramName).size();
  uint64_t* ints = new uint64_t[size];

  for (size_t i = 0; i < size; ++i)
    ints[i] = CLI::GetParam<std::vector<int>>(paramName)[i];

  return ints;
}

/**
 * Get the number of rows in a matrix parameter.
 */
size_t CLI_GetParamMatRows(const char* paramName)
{
  return CLI::GetParam<arma::mat>(paramName).n_rows;
}

/**
 * Get the number of columns in a matrix parameter.
 */
size_t CLI_GetParamMatCols(const char* paramName)
{
  return CLI::GetParam<arma::mat>(paramName).n_cols;
}

/**
 * Get the memory pointer for a matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CLI_GetParamMat(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::mat& mat = CLI::GetParam<arma::mat>(paramName);
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
size_t CLI_GetParamUMatRows(const char* paramName)
{
  return CLI::GetParam<arma::Mat<size_t>>(paramName).n_rows;
}

/**
 * Get the number of columns in an unsigned matrix parameter.
 */
size_t CLI_GetParamUMatCols(const char* paramName)
{
  return CLI::GetParam<arma::Mat<size_t>>(paramName).n_cols;
}

/**
 * Get the memory pointer for an unsigned matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CLI_GetParamUMat(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::Mat<size_t>& mat = CLI::GetParam<arma::Mat<size_t>>(paramName);
  if (mat.n_elem <= arma::arma_config::mat_prealloc)
  {
    // Copy the memory to something that we can give back to Julia.
    size_t* newMem = new size_t[mat.n_elem];
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
 * Get the number of rows in a column vector parameter.
 */
size_t CLI_GetParamColRows(const char* paramName)
{
  return CLI::GetParam<arma::vec>(paramName).n_rows;
}

/**
 * Get the memory pointer for a column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CLI_GetParamCol(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::vec& vec = CLI::GetParam<arma::vec>(paramName);
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
size_t CLI_GetParamUColRows(const char* paramName)
{
  return CLI::GetParam<arma::Col<size_t>>(paramName).n_rows;
}

/**
 * Get the memory pointer for an unsigned column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CLI_GetParamUCol(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::Col<size_t>& vec = CLI::GetParam<arma::Col<size_t>>(paramName);
  if (vec.n_elem <= arma::arma_config::mat_prealloc)
  {
    // Copy the memory to something we can give back to Julia.
    size_t* newMem = new size_t[vec.n_elem];
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
 * Get the number of columns in a row parameter.
 */
size_t CLI_GetParamRowCols(const char* paramName)
{
  return CLI::GetParam<arma::rowvec>(paramName).n_cols;
}

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* CLI_GetParamRow(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::rowvec& vec = CLI::GetParam<arma::rowvec>(paramName);
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
size_t CLI_GetParamURowCols(const char* paramName)
{
  return CLI::GetParam<arma::Row<size_t>>(paramName).n_cols;
}

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* CLI_GetParamURow(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::Row<size_t>& vec = CLI::GetParam<arma::Row<size_t>>(paramName);
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
size_t CLI_GetParamMatWithInfoRows(const char* paramName)
{
  return std::get<1>(CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)).n_rows;
}

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
size_t CLI_GetParamMatWithInfoCols(const char* paramName)
{
  return std::get<1>(CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)).n_cols;
}

/**
 * Get a pointer to an array of booleans representing whether or not dimensions
 * are categorical.  The calling function is expected to handle the memory
 * management.
 */
bool* CLI_GetParamMatWithInfoBoolPtr(const char* paramName)
{
  const data::DatasetInfo& d = std::get<0>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName));

  bool* dims = new bool[d.Dimensionality()];
  for (size_t i = 0; i < d.Dimensionality(); ++i)
    dims[i] = (d.Type(i) == data::Datatype::numeric) ? false : true;

  return dims;
}

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
double* CLI_GetParamMatWithInfoPtr(const char* paramName)
{
  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::mat& m = std::get<1>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName));
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
void CLI_EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

/**
 * Disable verbose output.
 */
void CLI_DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

/**
 * Reset the state of all timers.
 */
void CLI_ResetTimers()
{
  CLI::GetSingleton().timer.Reset();
}

/**
 * Set an argument as passed to the CLI object.
 */
void CLI_SetPassed(const char* paramName)
{
  CLI::SetPassed(paramName);
}

} // extern "C"
