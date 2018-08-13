/**
 * @file arma_util.cpp
 * @author Yasmine Dumouchel
 *
 * Utility function for Go to pass gonum object to an Armadillo Object and
 * vice versa.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "arma_util.h"
#include "arma_util.hpp"
#include "cli_util.hpp"
#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace util {

extern "C" {

/**
 * Pass Gonum Dense pointer and wrap an Armadillo mat around it.
 */
void MLPACK_ToArma_mat(const char *identifier, const double mat[], int row, int col)
{
  // Advanced constructor.
  arma::mat m(const_cast<double*>(mat), row, col, false, true);

  // Set input parameter with corresponding matrix in CLI.
  SetParam(identifier, m);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo rowvec around it.
 */
void MLPACK_ToArma_row(const char *identifier, const double rowvec[], int elem)
{
  // Advanced constructor.
  arma::rowvec m(const_cast<double*>(rowvec), elem, false, true);

  // Set input parameter with corresponding row in CLI.
  SetParam(identifier, m);
}

/**
 * Pass Gonum VecDense pointer and wrap an Armadillo colvec around it.
 */
void MLPACK_ToArma_col(const char *identifier, const double colvec[], int elem)
{
  // Advanced constructor.
  arma::colvec m(const_cast<double*>(colvec), elem, false, true);

  // Set input parameter with corresponding column in CLI.
  SetParam(identifier, m);
}

/**
 * Return the memory pointer of an Armadillo mat object.
 */
void *MLPACK_ArmaPtr_mat(const char *identifier)
{
  arma::mat output = CLI::GetParam<arma::mat>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  void *ptr = GetMemory(output);
  return ptr;
}

/**
 * Return the memory pointer of an Armadillo umat object.
 */
void *MLPACK_ArmaPtr_umat(const char *identifier)
{
  arma::Mat<double> output = arma::conv_to<arma::Mat<double>>::from(CLI::GetParam<arma::Mat<size_t>>(identifier));
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the memory pointer of an Armadillo row object.
 */
void *MLPACK_ArmaPtr_row(const char *identifier)
{
  arma::Row<double> output = CLI::GetParam<arma::Row<double>>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the memory pointer of an Armadillo urow object.
 */
void *MLPACK_ArmaPtr_urow(const char *identifier)
{
  arma::Row<double> output = arma::conv_to<arma::Row<double>>::from(CLI::GetParam<arma::Row<size_t>>(identifier));
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the memory pointer of an Armadillo col object.
 */
void *MLPACK_ArmaPtr_col(const char *identifier)
{
  arma::Col<double> output = CLI::GetParam<arma::Col<double>>(identifier);
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the memory pointer of an Armadillo ucol object.
 */
void *MLPACK_ArmaPtr_ucol(const char *identifier)
{
  arma::Col<double> output = arma::conv_to<arma::Col<double>>::from(CLI::GetParam<arma::Col<size_t>>(identifier));
  if (output.is_empty())
  {
    return NULL;
  }
  else
  {
  void *ptr = GetMemory(output);
  return ptr;
  }
}

/**
 * Return the number of rows in a Armadillo mat.
 */
int MLPACK_NumRow_mat(const char *identifier)
{
  arma::mat output = CLI::GetParam<arma::mat>(identifier);
  return output.n_rows;
}

/**
 * Return the number of columns in an Armadillo mat.
 */
int MLPACK_NumCol_mat(const char *identifier)
{
  arma::mat output = CLI::GetParam<arma::mat>(identifier);
  return output.n_cols;
}

/**
 * Return the number of elements in an Armadillo mat.
 */
int MLPACK_NumElem_mat(const char *identifier)
{
  arma::mat output = CLI::GetParam<arma::mat>(identifier);
  return output.n_elem;
}

/**
 * Return the number of rows in an Armadillo umat.
 */
int MLPACK_NumRow_umat(const char *identifier)
{
  arma::Mat<size_t>  output = CLI::GetParam<arma::Mat<size_t> >(identifier);
  return output.n_rows;
}

/**
 * Return the number of columns in an Armadillo umat.
 */
int MLPACK_NumCol_umat(const char *identifier)
{
  arma::Mat<size_t> output = CLI::GetParam<arma::Mat<size_t> >(identifier);
  return output.n_cols;
}

/**
 * Return the number of elements in an Armadillo umat.
 */
int MLPACK_NumElem_umat(const char *identifier)
{
  arma::Mat<size_t>  output = CLI::GetParam<arma::Mat<size_t> >(identifier);
  return output.n_elem;
}

/**
 * Return the number of elements in an Armadillo row.
 */
int MLPACK_NumElem_row(const char *identifier)
{
  arma::Row<double> output = CLI::GetParam<arma::Row<double>>(identifier);
  return output.n_elem;
}

/**
 * Return the number of elements in an Armadillo urow.
 */
int MLPACK_NumElem_urow(const char *identifier)
{
  arma::Row<size_t> output = CLI::GetParam<arma::Row<size_t>>(identifier);
  return output.n_elem;
}

/**
 * Return the number of elements in an Armadillo col.
 */
int MLPACK_NumElem_col(const char *identifier)
{
  arma::Col<double> output = CLI::GetParam<arma::Col<double>>(identifier);
  return output.n_elem;
}

/**
 * Return the number of elements in an Armadillo ucol.
 */
int MLPACK_NumElem_ucol(const char *identifier)
{
  arma::Col<size_t> output = CLI::GetParam<arma::Col<size_t>>(identifier);
  return output.n_elem;
}

} // extern "C"

} // namespace util
} // namespace mlpack
