/**
 * @file bindings/go/mlpack/capi/arma_util.cpp
 * @author Yasmine Dumouchel
 * @author Yashwant Singh
 *
 * Utility function for Go to pass gonum object to an Armadillo Object and
 * vice versa.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/bindings/go/mlpack/capi/arma_util.h>
#include "arma_util.hpp"
#include "io_util.hpp"
#include <mlpack/core/util/io.hpp>

namespace mlpack {
namespace util {

extern "C" {

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
void mlpackToArmaMat(const char* identifier, double* mat,
                     const size_t row, const size_t col)
{
  // Advanced constructor.
  arma::mat m(mat, row, col, false, true);

  // Set input parameter with corresponding matrix in CLI.
  SetParam(identifier, m);
}

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
void mlpackToArmaUmat(const char* identifier, double* mat,
                      const size_t row, const size_t col)
{
  // Advanced constructor.
  arma::mat m(mat, row, col, false, true);

  arma::Mat<size_t> matr = arma::conv_to<arma::Mat<size_t>>::from(m);

  // Set input parameter with corresponding matrix in CLI.
  SetParam(identifier, matr);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaRow(const char* identifier, double* rowvec, const size_t elem)
{
  // Advanced constructor.
  arma::rowvec m(rowvec, elem, false, true);

  // Set input parameter with corresponding row in CLI.
  SetParam(identifier, m);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaUrow(const char* identifier, double* rowvec, const size_t elem)
{
  // Advanced constructor.
  arma::rowvec m(rowvec, elem, false, true);

  arma::Row<size_t> matr = arma::conv_to<arma::Row<size_t>>::from(m);

  // Set input parameter with corresponding row in CLI.
  SetParam(identifier, matr);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void mlpackToArmaCol(const char* identifier, double* colvec, const size_t elem)
{
  // Advanced constructor.
  arma::colvec m(colvec, elem, false, true);

  // Set input parameter with corresponding column in CLI.
  SetParam(identifier, m);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void mlpackToArmaUcol(const char* identifier, double* colvec, const size_t elem)
{
  // Advanced constructor.
  arma::colvec m(colvec, elem, false, true);

  arma::Col<size_t> matr = arma::conv_to<arma::Col<size_t>>::from(m);

  // Set input parameter with corresponding column in CLI.
  SetParam(identifier, matr);
}
/**
 * Return the memory pointer of an Armadillo mat object.
 */
void* mlpackArmaPtrMat(const char* identifier)
{
  arma::mat& output = IO::GetParam<arma::mat>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  void* ptr = GetMemory(output);
  return ptr;
}

/**
 * Return the memory pointer of an Armadillo umat object.
 */
void* mlpackArmaPtrUmat(const char* identifier)
{
  arma::Mat<size_t>& m = IO::GetParam<arma::Mat<size_t>>(identifier);

  arma::mat output = arma::conv_to<arma::mat>::from(m);
  if (output.is_empty())
  {
    return NULL;
  }
  void* ptr = GetMemory(output);
  return ptr;
}

/**
 * Return the memory pointer of an Armadillo row object.
 */
void* mlpackArmaPtrRow(const char* identifier)
{
  arma::Row<double>& output = IO::GetParam<arma::Row<double>>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  void* ptr = GetMemory(output);
  return ptr;
}

/**
 * Return the memory pointer of an Armadillo urow object.
 */
void* mlpackArmaPtrUrow(const char* identifier)
{
  arma::Row<size_t>& m = IO::GetParam<arma::Row<size_t>>(identifier);

  arma::Row<double> output = arma::conv_to<arma::Row<double>>::from(m);
  if (output.is_empty())
  {
    return NULL;
  }
  void* ptr = GetMemory(output);
  return ptr;
}

/**
 * Return the memory pointer of an Armadillo col object.
 */
void* mlpackArmaPtrCol(const char* identifier)
{
  arma::Col<double>& output = IO::GetParam<arma::Col<double>>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  void* ptr = GetMemory(output);
  return ptr;
}

/**
 * Return the memory pointer of an Armadillo ucol object.
 */
void* mlpackArmaPtrUcol(const char* identifier)
{
  arma::Col<size_t>& m = IO::GetParam<arma::Col<size_t>>(identifier);

  arma::Col<double> output = arma::conv_to<arma::Col<double>>::from(m);
  if (output.is_empty())
  {
    return NULL;
  }
  void* ptr = GetMemory(output);
  return ptr;
}

/**
 * Return the number of rows in a Armadillo mat.
 */
int mlpackNumRowMat(const char* identifier)
{
  return IO::GetParam<arma::mat>(identifier).n_rows;
}

/**
 * Return the number of columns in an Armadillo mat.
 */
int mlpackNumColMat(const char* identifier)
{
  return IO::GetParam<arma::mat>(identifier).n_cols;
}

/**
 * Return the number of elements in an Armadillo mat.
 */
int mlpackNumElemMat(const char* identifier)
{
  return IO::GetParam<arma::mat>(identifier).n_elem;
}

/**
 * Return the number of rows in an Armadillo umat.
 */
int mlpackNumRowUmat(const char* identifier)
{
  return IO::GetParam<arma::Mat<size_t>>(identifier).n_rows;
}

/**
 * Return the number of columns in an Armadillo umat.
 */
int mlpackNumColUmat(const char* identifier)
{
  return IO::GetParam<arma::Mat<size_t>>(identifier).n_cols;
}

/**
 * Return the number of elements in an Armadillo umat.
 */
int mlpackNumElemUmat(const char* identifier)
{
  return IO::GetParam<arma::Mat<size_t>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo row.
 */
int mlpackNumElemRow(const char* identifier)
{
  return IO::GetParam<arma::Row<double>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo urow.
 */
int mlpackNumElemUrow(const char* identifier)
{
  return IO::GetParam<arma::Row<size_t>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo col.
 */
int mlpackNumElemCol(const char* identifier)
{
  return IO::GetParam<arma::Col<double>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo ucol.
 */
int mlpackNumElemUcol(const char* identifier)
{
  return IO::GetParam<arma::Col<size_t>>(identifier).n_elem;
}

/**
 * Call IO::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void mlpackToArmaMatWithInfo(const char* identifier,
                             const bool* dimensions,
                             double* memptr,
                             const size_t rows,
                             const size_t cols)
{
  data::DatasetInfo d(rows);
  for (size_t i = 0; i < d.Dimensionality(); ++i)
  {
    d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
        data::Datatype::numeric;
  }

  arma::mat m(memptr, rows, cols, false, true);
  std::get<0>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      identifier)) = std::move(d);
  std::get<1>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      identifier)) = std::move(m);
  IO::SetPassed(identifier);
}

/**
 * Get the number of elements in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoElements(const char* identifier)
{
  typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
  return std::get<1>(IO::GetParam<TupleType>(identifier)).n_elem;
}

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoRows(const char* identifier)
{
  typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
  return std::get<1>(IO::GetParam<TupleType>(identifier)).n_rows;
}

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoCols(const char* identifier)
{
  typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
  return std::get<1>(IO::GetParam<TupleType>(identifier)).n_cols;
}

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
void* mlpackArmaPtrMatWithInfoPtr(const char* identifier)
{
  typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
  arma::mat& m = std::get<1>(IO::GetParam<TupleType>(identifier));
  if (m.is_empty())
  {
    return NULL;
  }
  void* ptr = GetMemory(m);
  return ptr;
}

} // extern "C"

} // namespace util
} // namespace mlpack

