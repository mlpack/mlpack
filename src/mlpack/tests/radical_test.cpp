/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Test for RADICAL.
 *
 * This file is part of MLPACK 1.0.11.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/radical/radical.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

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
