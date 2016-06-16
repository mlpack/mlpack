/**
 * @file matrix_completion_test.cpp
 * @author Stephen Tu
 *
 * Tests for matrix completion
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/matrix_completion/matrix_completion.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::matrix_completion;

BOOST_AUTO_TEST_SUITE(MatrixCompletionTest);

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
BOOST_AUTO_TEST_CASE(UniformMatrixCompletionSDP)
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

  arma::mat recovered;
  MatrixCompletion mc(Xorig.n_rows, Xorig.n_cols, indices, values);
  mc.Recover(recovered);

  const double err =
    arma::norm(Xorig - recovered, "fro") /
    arma::norm(Xorig, "fro");
  BOOST_REQUIRE_SMALL(err, 1e-5);

  for (size_t i = 0; i < indices.n_cols; ++i)
  {
    BOOST_REQUIRE_CLOSE(
      recovered(indices(0, i), indices(1, i)),
      Xorig(indices(0, i), indices(1, i)),
      1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
