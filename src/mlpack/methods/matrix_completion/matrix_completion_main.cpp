/**
 * @file methods/matrix_completion/matrix_completion_main.cpp
 * @author Adarsh Santoria
 *
 * Executable for Matrix Completion.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME matrix_completion

#include <mlpack/core/util/mlpack_main.hpp>

#include "matrix_completion.hpp"

// Program Name.
BINDING_USER_NAME("Matrix Completion");

// Short description.
BINDING_SHORT_DESC(
    "A thin wrapper around nuclear norm minimization to solve low rank "
    "matrix completion problems.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of Matrix Completion.  "
    "This implements the popular nuclear norm minimization heuristic for "
    "low rank matrix completion problems and recover the recovery matrix. "
    "\n\n"
    "To work, this algorithm needs number of rows (specified by the " + 
    PRINT_PARAM_STRING("rows") + " parameter), number of columns (specified by "
    "the " + PRINT_PARAM_STRING("columns") + " parameter), indices  matrix of "
    "size (2 x p) (specified by the " + PRINT_PARAM_STRING("indices") +
    " parameter), values vector of length p (specified by the " +
    PRINT_PARAM_STRING("values") + " parameter).  "
    "\n\n"
    "Optionally, the maximum rank of solution (specified by the " + 
    PRINT_PARAM_STRING("rank") + " parameter) or initial_point matrix for the "
    "SDP optimization (specified by the " + PRINT_PARAM_STRING("initial") + 
    " parameter) can be used.  The recovered matrix containing the completed "
    "matrix will be saved with the " + PRINT_PARAM_STRING("recover") + " output "
    "parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to run Matrix Completion on the indices matrix " + 
    PRINT_DATASET("indices") + " with values vector " + PRINT_DATASET("values")
    + ", the following command could be used: "
    "\n\n" +
    PRINT_CALL("matrix_completion", "rows", "columns", "indices", "values"));

PARAM_INT_IN_REQ("rows", "Size of the neighborhood of similar users to "
    "consider for each query user.", "r");
PARAM_INT_IN_REQ("columns", "Rank of decomposed matrices (if 0, a heuristic is used to"
    " estimate the rank).", "c");
PARAM_UMATRIX_IN_REQ("indices", "Input dataset to run NCA on.", "i");
PARAM_COL_IN_REQ("values", "Labels for input dataset.", "v");

PARAM_INT_IN("rank", "Size of the neighborhood of similar users to "
    "consider for each query user.", "k", 1);
PARAM_MATRIX_IN("initial", "Input dataset to run NCA on.", "e");

PARAM_MATRIX_OUT("recover", "Matrix to output distances into.", "o");

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::util;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  size_t rows = (size_t) params.Get<int>("rows");
  size_t columns = (size_t) params.Get<int>("columns");

  // Load data.
  arma::Mat<size_t> indices = std::move(params.Get<arma::Mat<size_t>>("indices"));
  arma::vec values = std::move(params.Get<arma::vec>("values"));

  // Now create the MC object and recover the recovered matrix.
  timers.Start("mc_recovery");

  size_t rank = (size_t) params.Get<int>("rank");
  arma::mat initial = std::move(params.Get<arma::mat>("initial"));
  arma::mat recover;

  if (params.Has("rank"))
  {
    MatrixCompletion mc(rows, columns,indices, values, rank);
    mc.Recover(recover);
  }
  else if (params.Has("initial"))
  {
    MatrixCompletion mc(rows, columns,indices, values, initial);
    mc.Recover(recover);
  }
  else
  {
    MatrixCompletion mc(rows, columns,indices, values);
    mc.Recover(recover);
  }
  timers.Stop("mc_recovery");

  // Save the output.
  if (params.Has("recover"))
    params.Get<arma::mat>("recover") = std::move(recover);
}
