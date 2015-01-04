/**
 * @file matrix_completion_test.cpp
 * @author Stephen Tu
 *
 * Tests for matrix completion
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/matrix_completion/matrix_completion.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::matrix_completion;

BOOST_AUTO_TEST_SUITE(MatrixCompletionTest);

BOOST_AUTO_TEST_CASE(GaussianMatrixCompletionSDP)
{
  arma::mat Xorig, values;
  arma::umat indices;

  data::Load("completion_X.csv", Xorig, true, false);
  data::Load("completion_indices.csv", indices, true, false);

  values.set_size(indices.n_cols);
  for (size_t i = 0; i < indices.n_cols; ++i)
  {
    values(i) = Xorig(indices(0, i), indices(1, i));
  }

  MatrixCompletion mc(Xorig.n_rows, Xorig.n_cols, indices, values);
  mc.Recover();

  const double err =
    arma::norm(Xorig - mc.Recovered(), "fro") /
    arma::norm(Xorig, "fro");
  BOOST_REQUIRE_SMALL(err, 1e-5);

  for (size_t i = 0; i < indices.n_cols; ++i)
  {
    BOOST_REQUIRE_CLOSE(
      mc.Recovered()(indices(0, i), indices(1, i)),
      Xorig(indices(0, i), indices(1, i)),
      1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
