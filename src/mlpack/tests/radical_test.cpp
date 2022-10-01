/**
 * @file tests/radical_test.cpp
 * @author Nishant Mehta
 *
 * Test for RADICAL.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/radical.hpp>
#include "catch.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;

TEST_CASE("Radical_Test_Radical3D", "[RadicalTest]")
{
  mat matX;
  if (!data::Load("data_3d_mixed.txt", matX))
    FAIL("Cannot load dataset data_3d_mixed.txt");

  Radical rad(0.175, 5, 100, matX.n_rows - 1);

  mat matY;
  mat matW;
  rad.DoRadical(matX, matY, matW);

  mat matYT = trans(matY);
  double valEst = 0;

  for (uword i = 0; i < matYT.n_cols; ++i)
  {
    vec y = vec(matYT.col(i));
    valEst += rad.Vasicek(y);
  }

  mat matS;
  if (!data::Load("data_3d_ind.txt", matS))
    FAIL("Cannot load dataset data_3d_ind.txt");
  rad.DoRadical(matS, matY, matW);

  matYT = trans(matY);
  double valBest = 0;

  for (uword i = 0; i < matYT.n_cols; ++i)
  {
    vec y = vec(matYT.col(i));
    valBest += rad.Vasicek(y);
  }

  // Larger tolerance is sometimes needed.
  REQUIRE(valBest == Approx(valEst).epsilon(0.02));
}
