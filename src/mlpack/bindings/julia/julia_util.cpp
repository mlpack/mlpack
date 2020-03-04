/**
 * @file julia_util.cpp
 * @author Ryan Curtin
 *
 * Implementations of Julia binding functionality.
 */
#include <mlpack/bindings/julia/julia_util.h>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/cli.hpp>
#include <stdint.h>

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
 * Call CLI::SetParam<int>().  Julia always gives us an int64.
 */
void CLI_SetParamInt(const char* paramName, int64_t paramValue)
{
  CLI::GetParam<int>(paramName) = int(paramValue);
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
                              const uint64_t length)
{
  CLI::GetParam<std::vector<std::string>>(paramName).clear();
  CLI::GetParam<std::vector<std::string>>(paramName).resize(size_t(length));
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<std::vector<std::string>>() to set an individual element.
 */
void CLI_SetParamVectorStrStr(const char* paramName,
                              const char* str,
                              const uint64_t element)
{
  CLI::GetParam<std::vector<std::string>>(paramName)[size_t(element)] =
      std::string(str);
}

/**
 * Call CLI::SetParam<std::vector<int>>().  Julia always gives us int64s.
 */
void CLI_SetParamVectorInt(const char* paramName,
                           int64_t* ints,
                           const uint64_t length)
{
  // Create a std::vector<int> object; unfortunately this requires copying the
  // vector elements.
  std::vector<int> vec;
  vec.resize(size_t(length));
  for (size_t i = 0; i < (size_t) length; ++i)
    vec[i] = int(ints[i]);

  CLI::GetParam<std::vector<int>>(paramName) = std::move(vec);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::mat>().
 */
void CLI_SetParamMat(const char* paramName,
                     double* memptr,
                     const uint64_t rows,
                     const uint64_t cols,
                     const bool pointsAsRows)
{
  // Create the matrix as an alias.
  arma::mat m(memptr, arma::uword(rows), arma::uword(cols), false, true);
  CLI::GetParam<arma::mat>(paramName) = pointsAsRows ? m.t() : std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::Mat<size_t>>().
 */
void CLI_SetParamUMat(const char* paramName,
                      uint64_t* memptr,
                      const uint64_t rows,
                      const uint64_t cols,
                      const bool pointsAsRows)
{
  // If we're on a 64-bit system, we can create the matrix as an alias.
  if (sizeof(uint64_t) == sizeof(size_t))
  {
    // Create the matrix as an alias.
    arma::Mat<size_t> m((size_t*) memptr, arma::uword(rows), arma::uword(cols),
        false, true);
    CLI::GetParam<arma::Mat<size_t>>(paramName) = pointsAsRows ? m.t() :
        std::move(m);
    CLI::SetPassed(paramName);
  }
  else
  {
    // We have to perform conversion.  Create an alias of the memory we got, and
    // then convert it.
    arma::Mat<uint64_t> m(memptr, arma::uword(rows), arma::uword(cols), false,
        true);
    CLI::GetParam<arma::Mat<size_t>>(paramName) = pointsAsRows ?
        arma::conv_to<arma::Mat<size_t>>::from(m.t()) :
        arma::conv_to<arma::Mat<size_t>>::from(m);
    CLI::SetPassed(paramName);
  }
}

/**
 * Call CLI::SetParam<arma::rowvec>().
 */
void CLI_SetParamRow(const char* paramName,
                     double* memptr,
                     const uint64_t cols)
{
  arma::rowvec m(memptr, arma::uword(cols), false, true);
  CLI::GetParam<arma::rowvec>(paramName) = std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::Row<size_t>>().
 */
void CLI_SetParamURow(const char* paramName,
                      uint64_t* memptr,
                      const uint64_t cols)
{
  // If we're on a 64-bit system, we can create the matrix as an alias.
  if (sizeof(uint64_t) == sizeof(size_t))
  {
    arma::Row<size_t> m((size_t*) memptr, arma::uword(cols), false, true);
    CLI::GetParam<arma::Row<size_t>>(paramName) = std::move(m);
    CLI::SetPassed(paramName);
  }
  else
  {
    // We have to perform conversion.  Create an alias of the memory we got,
    // and then convert it.
    arma::Row<uint64_t> m(memptr, arma::uword(cols), false, true);
    CLI::GetParam<arma::Row<size_t>>(paramName) =
        arma::conv_to<arma::Row<size_t>>::from(m);
    CLI::SetPassed(paramName);
  }
}

/**
 * Call CLI::SetParam<arma::vec>().
 */
void CLI_SetParamCol(const char* paramName,
                     double* memptr,
                     const uint64_t rows)
{
  arma::vec m(memptr, arma::uword(rows), false, true);
  CLI::GetParam<arma::vec>(paramName) = std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::SetParam<arma::Row<size_t>>().
 */
void CLI_SetParamUCol(const char* paramName,
                      uint64_t* memptr,
                      const uint64_t rows)
{
  // If Julia gave us the right size, we can use an alias; otherwise we have to
  // copy.
  if (sizeof(uint64_t) == sizeof(size_t))
  {
    arma::Col<size_t> m((size_t*) memptr, arma::uword(rows), false, true);
    CLI::GetParam<arma::Col<size_t>>(paramName) = std::move(m);
    CLI::SetPassed(paramName);
  }
  else
  {
    arma::Col<uint64_t> m(memptr, arma::uword(rows), false, true);
    CLI::GetParam<arma::Col<size_t>>(paramName) =
        arma::conv_to<arma::Col<size_t>>::from(m);
    CLI::SetPassed(paramName);
  }
}

/**
 * Call CLI::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void CLI_SetParamMatWithInfo(const char* paramName,
                             bool* dimensions,
                             double* memptr,
                             const uint64_t rows,
                             const uint64_t cols,
                             const bool pointsAreRows)
{
  data::DatasetInfo d(pointsAreRows ? cols : rows);
  for (size_t i = 0; i < d.Dimensionality(); ++i)
  {
    d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
        data::Datatype::numeric;
  }

  arma::mat m(memptr, arma::uword(rows), arma::uword(cols), false, true);
  std::get<0>(CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = std::move(d);
  std::get<1>(CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = pointsAreRows ? std::move(m.t()) : std::move(m);
  CLI::SetPassed(paramName);
}

/**
 * Call CLI::GetParam<int>().
 */
int64_t CLI_GetParamInt(const char* paramName)
{
  return int64_t(CLI::GetParam<int>(paramName));
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
uint64_t CLI_GetParamVectorStrLen(const char* paramName)
{
  return uint64_t(CLI::GetParam<std::vector<std::string>>(paramName).size());
}

/**
 * Call CLI::GetParam<std::vector<std::string>>() and get the i'th string.
 */
const char* CLI_GetParamVectorStrStr(const char* paramName, const int64_t i)
{
  return CLI::GetParam<std::vector<std::string>>(paramName)[int(i)].c_str();
}

/**
 * Call CLI::GetParam<std::vector<int>>() and get the length of the vector.
 */
uint64_t CLI_GetParamVectorIntLen(const char* paramName)
{
  return uint64_t(CLI::GetParam<std::vector<int>>(paramName).size());
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
uint64_t CLI_GetParamMatRows(const char* paramName)
{
  return uint64_t(CLI::GetParam<arma::mat>(paramName).n_rows);
}

/**
 * Get the number of columns in a matrix parameter.
 */
uint64_t CLI_GetParamMatCols(const char* paramName)
{
  return uint64_t(CLI::GetParam<arma::mat>(paramName).n_cols);
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
uint64_t CLI_GetParamUMatRows(const char* paramName)
{
  return uint64_t(CLI::GetParam<arma::Mat<size_t>>(paramName).n_rows);
}

/**
 * Get the number of columns in an unsigned matrix parameter.
 */
uint64_t CLI_GetParamUMatCols(const char* paramName)
{
  return uint64_t(CLI::GetParam<arma::Mat<size_t>>(paramName).n_cols);
}

/**
 * Get the memory pointer for an unsigned matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
uint64_t* CLI_GetParamUMat(const char* paramName)
{
  arma::Mat<size_t>& mat = CLI::GetParam<arma::Mat<size_t>>(paramName);

  // Unfortunately, if size_t is not uint64_t, we will incur a copy.
  if (sizeof(uint64_t) == sizeof(size_t))
  {
    // Are we using preallocated memory?  If so we have to handle this more
    // carefully.
    if (mat.n_elem <= arma::arma_config::mat_prealloc)
    {
      // Copy the memory to something that we can give back to Julia.
      size_t* newMem = new size_t[mat.n_elem];
      arma::arrayops::copy(newMem, mat.mem, mat.n_elem);
      // We believe Julia will free it.  Hopefully we are right.
      return (uint64_t*) newMem;
    }
    else
    {
      arma::access::rw(mat.mem_state) = 1;
      return (uint64_t*) mat.memptr();
    }
  }
  else
  {
    uint64_t* newMem = new uint64_t[mat.n_elem];
    for (size_t i = 0; i < mat.n_elem; ++i)
      newMem[i] = uint64_t(mat[i]);
    // We believe Julia will free it.  Hopefully we are right.
    return newMem;
  }
}

/**
 * Get the number of rows in a column vector parameter.
 */
uint64_t CLI_GetParamColRows(const char* paramName)
{
  return uint64_t(CLI::GetParam<arma::vec>(paramName).n_rows);
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
uint64_t CLI_GetParamUColRows(const char* paramName)
{
  return uint64_t(CLI::GetParam<arma::Col<size_t>>(paramName).n_rows);
}

/**
 * Get the memory pointer for an unsigned column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
uint64_t* CLI_GetParamUCol(const char* paramName)
{
  arma::Col<size_t>& vec = CLI::GetParam<arma::Col<size_t>>(paramName);

  // If size_t is not the uint64_t that Julia expects, then unfortunately we
  // will have to make a copy.
  if (sizeof(uint64_t) == sizeof(size_t))
  {
    // Are we using preallocated memory?  If so we have to handle this more
    // carefully.
    if (vec.n_elem <= arma::arma_config::mat_prealloc)
    {
      // Copy the memory to something we can give back to Julia.
      size_t* newMem = new size_t[vec.n_elem];
      arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
      // We believe Julia will free it.  Hopefully we are right.
      return (uint64_t*) newMem;
    }
    else
    {
      arma::access::rw(vec.mem_state) = 1;
      return (uint64_t*) vec.memptr();
    }
  }
  else
  {
    uint64_t* newMem = new uint64_t[vec.n_elem];
    for (size_t i = 0; i < vec.n_elem; ++i)
      newMem[i] = uint64_t(vec[i]);
    // We believe Julia will free it.  Hopefully we are right.
    return newMem;
  }
}

/**
 * Get the number of columns in a row parameter.
 */
uint64_t CLI_GetParamRowCols(const char* paramName)
{
  return uint64_t(CLI::GetParam<arma::rowvec>(paramName).n_cols);
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
uint64_t CLI_GetParamURowCols(const char* paramName)
{
  return uint64_t(CLI::GetParam<arma::Row<size_t>>(paramName).n_cols);
}

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
uint64_t* CLI_GetParamURow(const char* paramName)
{
  arma::Row<size_t>& vec = CLI::GetParam<arma::Row<size_t>>(paramName);

  // If size_t is not the uint64_t that Julia expects, then unfortunately we
  // will have to make a copy.
  if (sizeof(size_t) == sizeof(uint64_t))
  {
    // Are we using preallocated memory?  If so we have to handle this more
    // carefully.
    if (vec.n_elem <= arma::arma_config::mat_prealloc)
    {
      // Copy the memory to something we can give back to Julia.
      size_t* newMem = new size_t[vec.n_elem];
      arma::arrayops::copy(newMem, vec.mem, vec.n_elem);
      return (uint64_t*) newMem;
    }
    else
    {
      arma::access::rw(vec.mem_state) = 1;
      return (uint64_t*) vec.memptr();
    }
  }
  else
  {
    uint64_t* newMem = new uint64_t[vec.n_elem];
    for (size_t i = 0; i < vec.n_elem; ++i)
      newMem[i] = uint64_t(vec[i]);
    // We believe that Julia will free the memory.  Hopefully we are right.
    return newMem;
  }
}

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
uint64_t CLI_GetParamMatWithInfoRows(const char* paramName)
{
  return uint64_t(std::get<1>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)).n_rows);
}

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
uint64_t CLI_GetParamMatWithInfoCols(const char* paramName)
{
  return uint64_t(std::get<1>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)).n_cols);
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
