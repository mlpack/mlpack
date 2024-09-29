/**
 * @file tests/matrix_completion_test.cpp
 * @author Stephen Tu
 *
 * Tests for matrix completion
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/matrix_completion.hpp>

#include "catch.hpp"

using namespace mlpack;

/**
 * A matrix completion test.
 *
 * The matrix X = F1 F2^T was generated such that the entries of Fi were iid
 * from the uniform distribution on [0, 1]. Then, enough random samples
 * (without replacement) were taking from X such that exact recovered was
 * possible.
 *
 * X is stored in the file "completion_X.csv" and the indices are stored in the
 * file "completion_indices.csv". Recovery was verified by solving the SDP with
 * Mosek.
 */
TEST_CASE("UniformMatrixCompletionSDP", "[MatrixCompletionTest]")
{
  arma::mat Xorig, values;
  arma::umat indices;

  if (!data::Load("completion_X.csv", Xorig, false, false))
    FAIL("Cannot load dataset completion_X.csv");
  if (!data::Load("completion_indices.csv", indices, false, false))
    FAIL("Cannot load dataset completion_indices.csv");

  values.set_size(indices.n_cols);
  for (size_t i = 0; i < indices.n_cols; ++i)
  {
    values(i) = Xorig(indices(0, i), indices(1, i));
  }

  arma::mat recovered;
  MatrixCompletion mc(Xorig.n_rows, Xorig.n_cols, indices, values);
  mc.Recover(recovered);

  const double err =
    arma::norm(Xorig - recovered, "fro") /
    arma::norm(Xorig, "fro");
  REQUIRE(err == Approx(0.0).margin(1e-5));

  for (size_t i = 0; i < indices.n_cols; ++i)
  {
    REQUIRE(recovered(indices(0, i), indices(1, i)) ==
       Approx(Xorig(indices(0, i), indices(1, i))).epsilon(1e-7));
  }
}
