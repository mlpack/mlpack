#include "arma_util.h"
#include "arma_util.hpp"
#include "cli_util.hpp"
#include <mlpack/core/util/cli.hpp>

namespace mlpack {

namespace util {

extern "C" {

void MLPACK_ToArma_mat(const char *identifier, const double mat[], int row, int col)
{
  // Advanced constructor.
  arma::mat m(const_cast<double*>(mat), row, col, false, true);

  // Set input parameter with corresponding matrix in CLI.
  SetParam(identifier, m);
}

void MLPACK_ToArma_row(const char *identifier, const double rowvec[], int elem)
{
  // Advanced constructor.
  arma::rowvec m(const_cast<double*>(rowvec), elem, false, true);

  // Set input parameter with corresponding matrix in CLI.
  SetParam(identifier, m);
}

void MLPACK_ToArma_col(const char *identifier, const double colvec[], int elem)
{
  // Advanced constructor.
  arma::colvec m(const_cast<double*>(colvec), elem, false, true);

  // Set input parameter with corresponding matrix in CLI.
  SetParam(identifier, m);
}

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


// Return the number of rows.
int MLPACK_NumRow_mat(const char *identifier)
{
  arma::mat output = CLI::GetParam<arma::mat>(identifier);
  return output.n_rows;
}

// Return the number of cols.
int MLPACK_NumCol_mat(const char *identifier)
{
  arma::mat output = CLI::GetParam<arma::mat>(identifier);
  return output.n_cols;
}

// Return the number of elems.
int MLPACK_NumElem_mat(const char *identifier)
{
  arma::mat output = CLI::GetParam<arma::mat>(identifier);
  return output.n_elem;
}

// Return the number of rows.
int MLPACK_NumRow_umat(const char *identifier)
{
  arma::Mat<size_t>  output = CLI::GetParam<arma::Mat<size_t> >(identifier);
  return output.n_rows;
}

// Return the number of cols.
int MLPACK_NumCol_umat(const char *identifier)
{
  arma::Mat<size_t> output = CLI::GetParam<arma::Mat<size_t> >(identifier);
  return output.n_cols;
}

// Return the number of elems.
int MLPACK_NumElem_umat(const char *identifier)
{
  arma::Mat<size_t>  output = CLI::GetParam<arma::Mat<size_t> >(identifier);
  return output.n_elem;
}

int MLPACK_Size_urow(const char *identifier)
{
  arma::Row<size_t> output = CLI::GetParam<arma::Row<size_t>>(identifier);
  return output.n_rows;
}

int MLPACK_Size_row(const char *identifier)
{
  arma::Row<double> output = CLI::GetParam<arma::Row<double>>(identifier);
  return output.n_rows;
}

// Return the number of elems.
int MLPACK_NumElem_urow(const char *identifier)
{
  arma::Row<size_t> output = CLI::GetParam<arma::Row<size_t>>(identifier);
  return output.n_elem;
}

// Return the number of elems.
int MLPACK_NumElem_row(const char *identifier)
{
  arma::Row<double> output = CLI::GetParam<arma::Row<double>>(identifier);
  return output.n_elem;
}

// Return the number of cols.
int MLPACK_Size_ucol(const char *identifier)
{
  arma::Col<size_t> output = CLI::GetParam<arma::Col<size_t>>(identifier);
  return output.n_cols;
}

// Return the number of cols.
int MLPACK_Size_col(const char *identifier)
{
  arma::Col<double> output = CLI::GetParam<arma::Col<double>>(identifier);
  return output.n_cols;
}

// Return the number of elems.
int MLPACK_NumElem_ucol(const char *identifier)
{
  arma::Col<size_t> output = CLI::GetParam<arma::Col<size_t>>(identifier);
  return output.n_elem;
}

// Return the number of elems.
int MLPACK_NumElem_col(const char *identifier)
{
  arma::Col<double> output = CLI::GetParam<arma::Col<double>>(identifier);
  return output.n_elem;
}

} // extern "C"

} // namespace util

} // namespace mlpack
