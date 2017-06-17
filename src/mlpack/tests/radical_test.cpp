/**
 * @file radical_main.cpp
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
#include <mlpack/methods/radical/radical.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

BOOST_AUTO_TEST_SUITE(RadicalTest);

using namespace mlpack;
using namespace mlpack::radical;
using namespace std;
using namespace arma;

BOOST_AUTO_TEST_CASE(Radical_Test_Radical3D)
{
  mat matX;
  data::Load("data_3d_mixed.txt", matX);

  Radical rad(0.175, 5, 100, matX.n_rows - 1);

  mat matY;
  mat matW;
  rad.DoRadical(matX, matY, matW);

  mat matYT = trans(matY);
  double valEst = 0;

  for (uword i = 0; i < matYT.n_cols; i++)
  {
    vec y = vec(matYT.col(i));
    valEst += rad.Vasicek(y);
  }

  mat matS;
  data::Load("data_3d_ind.txt", matS);
  rad.DoRadical(matS, matY, matW);

  matYT = trans(matY);
  double valBest = 0;

  for (uword i = 0; i < matYT.n_cols; i++)
  {
    vec y = vec(matYT.col(i));
    valBest += rad.Vasicek(y);
  }

  BOOST_REQUIRE_CLOSE(valBest, valEst, 0.25);
}

BOOST_AUTO_TEST_SUITE_END();
