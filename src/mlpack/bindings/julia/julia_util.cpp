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
#include <mlpack/core/util/io.hpp>
#include <stdint.h>

using namespace mlpack;

extern "C" {

/**
 * Get a new util::Params object, encoded as a stack-allocated void pointer.
 * You are responsible for freeing this!
 */
void* GetParameters(const char* bindingName)
{
  util::Params* p = new util::Params(IO::Parameters(bindingName));
  return (void*) p;
}

/**
 * Delete a util::Params object that has been encoded as a void pointer.
 */
void DeleteParameters(void* in)
{
  util::Params* p = (util::Params*) in;
  delete p;
}

/**
 * Get a new util::Timers object, encoded as a heap-allocated void pointer.  You
 * are responsible for freeing this!  You can use `DeleteTimers(void*)`.
 */
void* Timers()
{
  util::Timers* t = new util::Timers();
  return (void*) t;
}

/**
 * Delete a util::Timers object that has been encoded as a void pointer.
 */
void DeleteTimers(void* in)
{
  util::Timers* t = (util::Timers*) in;
  delete t;
}

/**
 * Call params.SetParam<int>().
 */
void SetParamInt(void* params, const char* paramName, int paramValue)
{
  util::Params* p = (util::Params*) params;
  p->Get<int>(paramName) = paramValue;
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<double>().
 */
void SetParamDouble(void* params, const char* paramName, double paramValue)
{
  util::Params* p = (util::Params*) params;
  p->Get<double>(paramName) = paramValue;
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<std::string>().
 */
void SetParamString(void* params, const char* paramName, const char* paramValue)
{
  util::Params* p = (util::Params*) params;
  p->Get<std::string>(paramName) = paramValue;
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<bool>().
 */
void SetParamBool(void* params, const char* paramName, bool paramValue)
{
  util::Params* p = (util::Params*) params;
  p->Get<bool>(paramName) = paramValue;
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<std::vector<std::string>>() to set the length.
 */
void SetParamVectorStrLen(void* params,
                          const char* paramName,
                          const size_t length)
{
  util::Params* p = (util::Params*) params;
  p->Get<std::vector<std::string>>(paramName).clear();
  p->Get<std::vector<std::string>>(paramName).resize(length);
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<std::vector<std::string>>() to set an individual
 * element.
 */
void SetParamVectorStrStr(void* params,
                          const char* paramName,
                          const char* str,
                          const size_t element)
{
  util::Params* p = (util::Params*) params;
  p->Get<std::vector<std::string>>(paramName)[element] =
      std::string(str);
}

/**
 * Call params.SetParam<std::vector<int>>().
 */
void SetParamVectorInt(void* params,
                       const char* paramName,
                       long long* ints,
                       const size_t length)
{
  util::Params* p = (util::Params*) params;

  // Create a std::vector<int> object; unfortunately this requires copying the
  // vector elements.
  std::vector<int> vec;
  vec.resize(length);
  for (size_t i = 0; i < length; ++i)
    vec[i] = int(ints[i]);

  p->Get<std::vector<int>>(paramName) = std::move(vec);
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<arma::mat>().
 */
void SetParamMat(void* params,
                 const char* paramName,
                 double* memptr,
                 const size_t rows,
                 const size_t cols,
                 const bool pointsAsRows)
{
  util::Params* p = (util::Params*) params;

  // Create the matrix as an alias.
  arma::mat m(memptr, arma::uword(rows), arma::uword(cols), false, false);
  p->Get<arma::mat>(paramName) = pointsAsRows ? m.t() : std::move(m);
  p->SetPassed(paramName);
}

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
                  const bool pointsAsRows)
{
  util::Params* p = (util::Params*) params;

  // Create the matrix as an alias.
  arma::Mat<long long> m(memptr, arma::uword(rows), arma::uword(cols), true,
      true);
  arma::Mat<size_t> convM = arma::conv_to<arma::Mat<size_t>>::from(m - 1);
  p->Get<arma::Mat<size_t>>(paramName) = pointsAsRows ? convM.t() :
      std::move(convM);
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<arma::rowvec>().
 */
void SetParamRow(void* params,
                 const char* paramName,
                 double* memptr,
                 const size_t cols)
{
  util::Params* p = (util::Params*) params;
  arma::rowvec m(memptr, arma::uword(cols), false, false);
  p->Get<arma::rowvec>(paramName) = std::move(m);
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<arma::Row<size_t>>().
 */
void SetParamURow(void* params,
                  const char* paramName,
                  long long* memptr,
                  const size_t cols)
{
  util::Params* p = (util::Params*) params;
  arma::Row<long long> m(memptr, arma::uword(cols), false, true);
  arma::Row<size_t> convM = arma::conv_to<arma::Row<size_t>>::from(m - 1);
  p->Get<arma::Row<size_t>>(paramName) = std::move(convM);
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<arma::vec>().
 */
void SetParamCol(void* params,
                 const char* paramName,
                 double* memptr,
                 const size_t rows)
{
  util::Params* p = (util::Params*) params;
  arma::vec m(memptr, arma::uword(rows), false, false);
  p->Get<arma::vec>(paramName) = std::move(m);
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<arma::Row<size_t>>().
 */
void SetParamUCol(void* params,
                  const char* paramName,
                  long long* memptr,
                  const size_t rows)
{
  util::Params* p = (util::Params*) params;
  arma::Col<long long> m(memptr, arma::uword(rows), false, true);
  arma::Col<size_t> convM = arma::conv_to<arma::Col<size_t>>::from(m - 1);
  p->Get<arma::Col<size_t>>(paramName) = std::move(convM);
  p->SetPassed(paramName);
}

/**
 * Call params.SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void SetParamMatWithInfo(void* params,
                         const char* paramName,
                         bool* dimensions,
                         double* memptr,
                         const size_t rows,
                         const size_t cols,
                         const bool pointsAreRows)
{
  util::Params* p = (util::Params*) params;
  data::DatasetInfo d(pointsAreRows ? cols : rows);
  bool hasCategoricals = false;
  for (size_t i = 0; i < d.Dimensionality(); ++i)
  {
    d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
        data::Datatype::numeric;
    if (dimensions[i])
      hasCategoricals = true;
  }

  arma::mat alias(memptr, arma::uword(rows), arma::uword(cols), false, false);
  arma::mat m = pointsAreRows ? alias.t() : alias;

  // Do we need to find how many categories we have?
  if (hasCategoricals)
  {
    // Compute the maximum in each dimension.
    const arma::vec maxs = arma::max(m, 1);

    for (size_t i = 0; i < d.Dimensionality(); ++i)
    {
      if (dimensions[i])
      {
        // Map the right number of objects.
        for (size_t j = 1; j <= (size_t) maxs[i]; ++j)
        {
          std::ostringstream oss;
          oss << j;
          d.MapString<double>(oss.str(), i);
        }

        // In Julia we specify the categorical value from 1 to the number of
        // categories, but in C++ we expect 0 to the number of categories minus
        // one.  (Just like the labels.)
        m.row(i) -= 1.0;
      }
    }
  }

  std::get<0>(p->Get<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = std::move(d);
  std::get<1>(p->Get<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = std::move(m);
  p->SetPassed(paramName);
}

/**
 * Call params.GetParam<int>().
 */
int GetParamInt(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<int>(paramName);
}

/**
 * Call params.GetParam<double>().
 */
double GetParamDouble(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<double>(paramName);
}

/**
 * Call params.GetParam<std::string>().
 */
const char* GetParamString(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<std::string>(paramName).c_str();
}

/**
 * Call params.GetParam<bool>().
 */
bool GetParamBool(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<bool>(paramName);
}

/**
 * Call params.GetParam<std::vector<std::string>>() and get the length of the
 * vector.
 */
size_t GetParamVectorStrLen(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<std::vector<std::string>>(paramName).size();
}

/**
 * Call params.GetParam<std::vector<std::string>>() and get the i'th string.
 */
const char* GetParamVectorStrStr(void* params,
                                 const char* paramName,
                                 const size_t i)
{
  util::Params* p = (util::Params*) params;
  return p->Get<std::vector<std::string>>(paramName)[i].c_str();
}

/**
 * Call params.GetParam<std::vector<int>>() and get the length of the vector.
 */
size_t GetParamVectorIntLen(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<std::vector<int>>(paramName).size();
}

/**
 * Call params.GetParam<std::vector<int>>() and return a pointer to the vector.
 * The vector will be created in-place and it is expected that the calling
 * function will take ownership.
 */
long long* GetParamVectorIntPtr(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  const size_t size = p->Get<std::vector<int>>(paramName).size();
  if (size == 0)
    return NULL;

  long long* ints = new long long[size];

  for (size_t i = 0; i < size; ++i)
    ints[i] = (long long)(p->Get<std::vector<int>>(paramName)[i]);

  return ints;
}

/**
 * Get the number of rows in a matrix parameter.
 */
size_t GetParamMatRows(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<arma::mat>(paramName).n_rows;
}

/**
 * Get the number of columns in a matrix parameter.
 */
size_t GetParamMatCols(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<arma::mat>(paramName).n_cols;
}

/**
 * Get the memory pointer for a matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* GetParamMat(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::mat& mat = p->Get<arma::mat>(paramName);
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
    #if ARMA_VERSION_MAJOR >= 10
      arma::access::rw(mat.n_alloc) = 0;
    #endif
    return mat.memptr();
  }
}

/**
 * Get the number of rows in an unsigned matrix parameter.
 */
size_t GetParamUMatRows(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<arma::Mat<size_t>>(paramName).n_rows;
}

/**
 * Get the number of columns in an unsigned matrix parameter.
 */
size_t GetParamUMatCols(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<arma::Mat<size_t>>(paramName).n_cols;
}

/**
 * Get the memory pointer for an unsigned matrix parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* GetParamUMat(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  arma::Mat<size_t>& mat = p->Get<arma::Mat<size_t>>(paramName);

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
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
    #if ARMA_VERSION_MAJOR >= 10
      arma::access::rw(mat.n_alloc) = 0;
    #endif
    return mat.memptr();
  }
}

/**
 * Get the number of rows in a column vector parameter.
 */
size_t GetParamColRows(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<arma::vec>(paramName).n_rows;
}

/**
 * Get the memory pointer for a column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* GetParamCol(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::vec& vec = p->Get<arma::vec>(paramName);
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
    #if ARMA_VERSION_MAJOR >= 10
      arma::access::rw(vec.n_alloc) = 0;
    #endif
    return vec.memptr();
  }
}

/**
 * Get the number of columns in an unsigned column vector parameter.
 */
size_t GetParamUColRows(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<arma::Col<size_t>>(paramName).n_rows;
}

/**
 * Get the memory pointer for an unsigned column vector parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* GetParamUCol(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;

  arma::Col<size_t>& vec = p->Get<arma::Col<size_t>>(paramName);

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
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
    #if ARMA_VERSION_MAJOR >= 10
      arma::access::rw(vec.n_alloc) = 0;
    #endif
    return vec.memptr();
  }
}

/**
 * Get the number of columns in a row parameter.
 */
size_t GetParamRowCols(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<arma::rowvec>(paramName).n_cols;
}

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
double* GetParamRow(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  arma::rowvec& vec = p->Get<arma::rowvec>(paramName);
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
    #if ARMA_VERSION_MAJOR >= 10
      arma::access::rw(vec.n_alloc) = 0;
    #endif
    return vec.memptr();
  }
}

/**
 * Get the number of columns in a row parameter.
 */
size_t GetParamURowCols(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return p->Get<arma::Row<size_t>>(paramName).n_cols;
}

/**
 * Get the memory pointer for a row parameter.
 * Note that this will assume that whatever is calling will take ownership of
 * the memory!
 */
size_t* GetParamURow(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;

  arma::Row<size_t>& vec = p->Get<arma::Row<size_t>>(paramName);

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
    #if ARMA_VERSION_MAJOR >= 10
      arma::access::rw(vec.n_alloc) = 0;
    #endif
    return vec.memptr();
  }
}

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
size_t GetParamMatWithInfoRows(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return std::get<1>(p->Get<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)).n_rows;
}

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
size_t GetParamMatWithInfoCols(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  return std::get<1>(p->Get<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)).n_cols;
}

/**
 * Get a pointer to an array of booleans representing whether or not dimensions
 * are categorical.  The calling function is expected to handle the memory
 * management.
 */
bool* GetParamMatWithInfoBoolPtr(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;

  const data::DatasetInfo& d = std::get<0>(
      p->Get<std::tuple<data::DatasetInfo, arma::mat>>(paramName));

  bool* dims = new bool[d.Dimensionality()];
  for (size_t i = 0; i < d.Dimensionality(); ++i)
    dims[i] = (d.Type(i) == data::Datatype::numeric) ? false : true;

  return dims;
}

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
double* GetParamMatWithInfoPtr(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;

  // Are we using preallocated memory?  If so we have to handle this more
  // carefully.
  data::DatasetInfo& di = std::get<0>(
      p->Get<std::tuple<data::DatasetInfo, arma::mat>>(paramName));
  arma::mat& m = std::get<1>(
      p->Get<std::tuple<data::DatasetInfo, arma::mat>>(paramName));

  // Add 1 to any categorical columns, to map them back to the 1-indexed values
  // expected in Julia.
  for (size_t d = 0; d < di.Dimensionality(); ++d)
  {
    if (di.Type(d) == data::Datatype::categorical)
      m.row(d) += 1.0;
  }

  if (m.n_elem <= arma::arma_config::mat_prealloc)
  {
    double* newMem = new double[m.n_elem];
    arma::arrayops::copy(newMem, m.mem, m.n_elem);
    return newMem;
  }
  else
  {
    arma::access::rw(m.mem_state) = 1;
    #if ARMA_VERSION_MAJOR >= 10
      arma::access::rw(m.n_alloc) = 0;
    #endif
    return m.memptr();
  }
}

/**
 * Enable verbose output.
 */
void EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

/**
 * Disable verbose output.
 */
void DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

/**
 * Set an argument as passed to the IO object.
 */
void SetPassed(void* params, const char* paramName)
{
  util::Params* p = (util::Params*) params;
  p->SetPassed(paramName);
}

} // extern "C"
