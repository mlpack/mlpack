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
void mlpackToArmaMat(void* params,
                     const char* identifier,
                     double* mat,
                     const size_t row,
                     const size_t col,
                     bool transpose)
{
  util::Params& p = *((util::Params*) params);

  // Advanced constructor.
  arma::mat m(mat, row, col, false, false);

  // Transpose if necessary.
  if (transpose)
    arma::inplace_trans(m);

  // Set input parameter with corresponding matrix in IO.
  SetParam(p, identifier, m);
}

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
void mlpackToArmaUmat(void* params,
                      const char* identifier,
                      double* mat,
                      const size_t row,
                      const size_t col)
{
  util::Params& p = *((util::Params*) params);

  // Advanced constructor.
  arma::mat m(mat, row, col, false, false);

  arma::Mat<size_t> matr = arma::conv_to<arma::Mat<size_t>>::from(m);

  // Set input parameter with corresponding matrix in IO.
  SetParam(p, identifier, matr);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaRow(void* params,
                     const char* identifier,
                     double* rowvec,
                     const size_t elem)
{
  util::Params& p = *((util::Params*) params);

  // Advanced constructor.
  arma::rowvec m(rowvec, elem, false, false);

  // Set input parameter with corresponding row in IO.
  SetParam(p, identifier, m);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void mlpackToArmaUrow(void* params,
                      const char* identifier,
                      double* rowvec,
                      const size_t elem)
{
  util::Params& p = *((util::Params*) params);

  // Advanced constructor.
  arma::rowvec m(rowvec, elem, false, false);

  arma::Row<size_t> matr = arma::conv_to<arma::Row<size_t>>::from(m);

  // Set input parameter with corresponding row in IO.
  SetParam(p, identifier, matr);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void mlpackToArmaCol(void* params,
                     const char* identifier,
                     double* colvec,
                     const size_t elem)
{
  util::Params& p = *((util::Params*) params);

  // Advanced constructor.
  arma::colvec m(colvec, elem, false, false);

  // Set input parameter with corresponding column in IO.
  SetParam(p, identifier, m);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void mlpackToArmaUcol(void* params,
                      const char* identifier,
                      double* colvec,
                      const size_t elem)
{
  util::Params& p = *((util::Params*) params);

  // Advanced constructor.
  arma::colvec m(colvec, elem, false, false);

  arma::Col<size_t> matr = arma::conv_to<arma::Col<size_t>>::from(m);

  // Set input parameter with corresponding column in IO.
  SetParam(p, identifier, matr);
}
/**
 * Return the memory pointer of an Armadillo mat object.
 */
void* mlpackArmaPtrMat(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  arma::mat& output = p.Get<arma::mat>(identifier);
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
void* mlpackArmaPtrUmat(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  arma::Mat<size_t>& m = p.Get<arma::Mat<size_t>>(identifier);

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
void* mlpackArmaPtrRow(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  arma::Row<double>& output = p.Get<arma::Row<double>>(identifier);
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
void* mlpackArmaPtrUrow(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  arma::Row<size_t>& m = p.Get<arma::Row<size_t>>(identifier);

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
void* mlpackArmaPtrCol(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  arma::Col<double>& output = p.Get<arma::Col<double>>(identifier);
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
void* mlpackArmaPtrUcol(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  arma::Col<size_t>& m = p.Get<arma::Col<size_t>>(identifier);

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
int mlpackNumRowMat(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::mat>(identifier).n_rows;
}

/**
 * Return the number of columns in an Armadillo mat.
 */
int mlpackNumColMat(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::mat>(identifier).n_cols;
}

/**
 * Return the number of elements in an Armadillo mat.
 */
int mlpackNumElemMat(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::mat>(identifier).n_elem;
}

/**
 * Return the number of rows in an Armadillo umat.
 */
int mlpackNumRowUmat(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::Mat<size_t>>(identifier).n_rows;
}

/**
 * Return the number of columns in an Armadillo umat.
 */
int mlpackNumColUmat(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::Mat<size_t>>(identifier).n_cols;
}

/**
 * Return the number of elements in an Armadillo umat.
 */
int mlpackNumElemUmat(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::Mat<size_t>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo row.
 */
int mlpackNumElemRow(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::Row<double>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo urow.
 */
int mlpackNumElemUrow(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::Row<size_t>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo col.
 */
int mlpackNumElemCol(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::Col<double>>(identifier).n_elem;
}

/**
 * Return the number of elements in an Armadillo ucol.
 */
int mlpackNumElemUcol(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  return p.Get<arma::Col<size_t>>(identifier).n_elem;
}

/**
 * Call IO::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
 */
void mlpackToArmaMatWithInfo(void* params,
                             const char* identifier,
                             const bool* dimensions,
                             double* memptr,
                             const size_t rows,
                             const size_t cols)
{
  util::Params& p = *((util::Params*) params);
  data::DatasetInfo d(rows);
  bool hasCategoricals = false;
  for (size_t i = 0; i < d.Dimensionality(); ++i)
  {
    d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
        data::Datatype::numeric;
    if (dimensions[i])
      hasCategoricals = true;
  }

  arma::mat m(memptr, rows, cols, false, false);

  // Do we need to find how many categories we have?
  if (hasCategoricals)
  {
    arma::vec maxs = arma::max(m, 1) + 1;

    for (size_t i = 0; i < d.Dimensionality(); ++i)
    {
      if (dimensions[i])
      {
        // Map the right number of objects.
        for (size_t j = 0; j < (size_t) maxs[i]; ++j)
        {
          std::ostringstream oss;
          oss << j;
          d.MapString<double>(oss.str(), i);
        }
      }
    }
  }

  std::get<0>(p.Get<std::tuple<data::DatasetInfo, arma::mat>>(identifier)) =
      std::move(d);
  std::get<1>(p.Get<std::tuple<data::DatasetInfo, arma::mat>>(identifier)) =
      std::move(m);
  p.SetPassed(identifier);
}

/**
 * Get the number of elements in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoElements(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
  return std::get<1>(p.Get<TupleType>(identifier)).n_elem;
}

/**
 * Get the number of rows in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoRows(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
  return std::get<1>(p.Get<TupleType>(identifier)).n_rows;
}

/**
 * Get the number of columns in a matrix with DatasetInfo parameter.
 */
int mlpackArmaMatWithInfoCols(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
  return std::get<1>(p.Get<TupleType>(identifier)).n_cols;
}

/**
 * Get a pointer to the memory of the matrix.  The calling function is expected
 * to own the memory.
 */
void* mlpackArmaPtrMatWithInfoPtr(void* params, const char* identifier)
{
  util::Params& p = *((util::Params*) params);
  typedef std::tuple<data::DatasetInfo, arma::mat> TupleType;
  arma::mat& m = std::get<1>(p.Get<TupleType>(identifier));
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

