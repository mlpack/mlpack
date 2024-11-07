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

TEMPLATE_TEST_CASE("Radical_Test_Radical3D", "[RadicalTest]", float, double)
{
  using ElemType = TestType;
  using VecType = arma::Col<ElemType>;
  using MatType = arma::Mat<ElemType>;

  MatType matX;
  if (!data::Load("data_3d_mixed.txt", matX))
    FAIL("Cannot load dataset data_3d_mixed.txt");

  Radical rad(0.175, 5, 100, matX.n_rows - 1);

  MatType matY;
  MatType matW;
  rad.Apply(matX, matY, matW);

  const size_t m = std::floor(std::sqrt((ElemType) matX.n_rows));

  MatType matYT = trans(matY);
  ElemType valEst = 0;

  for (arma::uword i = 0; i < matYT.n_cols; ++i)
  {
    VecType y(matYT.col(i));
    valEst += rad.Vasicek(y, m);
  }

  MatType matS;
  if (!data::Load("data_3d_ind.txt", matS))
    FAIL("Cannot load dataset data_3d_ind.txt");
  rad.Apply(matS, matY, matW);

  matYT = trans(matY);
  ElemType valBest = 0;

  for (arma::uword i = 0; i < matYT.n_cols; ++i)
  {
    VecType y(matYT.col(i));
    valBest += rad.Vasicek(y, m);
  }

  // Larger tolerance is sometimes needed.
  REQUIRE(valBest == Approx(valEst).epsilon(0.02));
}
