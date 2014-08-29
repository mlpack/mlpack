#include <mlpack/core.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/update_rules/svd_batch_learning.hpp>
#include <mlpack/methods/amf/init_rules/random_init.hpp>
#include <mlpack/methods/amf/termination_policies/validation_RMSE_termination.hpp>
#include <mlpack/methods/amf/termination_policies/simple_tolerance_termination.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(SVDBatchTest);

using namespace std;
using namespace mlpack;
using namespace mlpack::amf;
using namespace arma;

/**
 * Make sure the SVD Batch lerning is converging.
 *
 * This file is part of MLPACK 1.0.10.
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
BOOST_AUTO_TEST_CASE(SVDBatchConvergenceElementTest)
{
  mlpack::math::RandomSeed(10);
  sp_mat data;
  data.sprandn(1000, 1000, 0.2);
  AMF<SimpleToleranceTermination<sp_mat>, 
      RandomInitialization, 
      SVDBatchLearning> amf;
  mat m1,m2;
  amf.Apply(data, 2, m1, m2);
  
  BOOST_REQUIRE_NE(amf.TerminationPolicy().Iteration(), 
                    amf.TerminationPolicy().MaxIterations());
}

/**
 * Make sure the momentum is working okay.
 */
BOOST_AUTO_TEST_CASE(SVDBatchMomentumTest)
{
  mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Generate list of locations for batch insert constructor for sparse
  // matrices.
  arma::umat locations(2, dataset.n_cols);
  arma::vec values(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    // We have to transpose it because items are rows, and users are columns.
    locations(0, i) = ((arma::uword) dataset(0, i));
    locations(1, i) = ((arma::uword) dataset(1, i));
    values(i) = dataset(2, i);
  }

  // Find maximum user and item IDs.
  const size_t maxUserID = (size_t) max(locations.row(0)) + 1;
  const size_t maxItemID = (size_t) max(locations.row(1)) + 1;

  // Fill sparse matrix.
  sp_mat cleanedData = arma::sp_mat(locations, values, maxUserID, maxItemID);

  mlpack::math::RandomSeed(10);
  ValidationRMSETermination<sp_mat> vrt(cleanedData, 2000);
  AMF<ValidationRMSETermination<sp_mat>,
      RandomInitialization,
      SVDBatchLearning> amf_1(vrt,
                              RandomInitialization(),
                              SVDBatchLearning(0.0009, 0, 0, 0));

  mat m1,m2;
  double RMSE_1 = amf_1.Apply(cleanedData, 2, m1, m2);
  size_t iter_1 = amf_1.TerminationPolicy().Iteration();

  mlpack::math::RandomSeed(10);
  AMF<ValidationRMSETermination<sp_mat>,
      RandomInitialization,
      SVDBatchLearning> amf_2(vrt,
                              RandomInitialization(),
                              SVDBatchLearning(0.0009, 0, 0, 0.8));

  double RMSE_2 = amf_2.Apply(cleanedData, 2, m1, m2);
  size_t iter_2 = amf_2.TerminationPolicy().Iteration();

  BOOST_REQUIRE_LE(RMSE_2, RMSE_1);
  BOOST_REQUIRE_LE(iter_2, iter_1);
}

/**
 * Make sure the regularization is working okay.
 */
BOOST_AUTO_TEST_CASE(SVDBatchRegularizationTest)
{
  mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Generate list of locations for batch insert constructor for sparse
  // matrices.
  arma::umat locations(2, dataset.n_cols);
  arma::vec values(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    // We have to transpose it because items are rows, and users are columns.
    locations(0, i) = ((arma::uword) dataset(0, i));
    locations(1, i) = ((arma::uword) dataset(1, i));
    values(i) = dataset(2, i);
  }

  // Find maximum user and item IDs.
  const size_t maxUserID = (size_t) max(locations.row(0)) + 1;
  const size_t maxItemID = (size_t) max(locations.row(1)) + 1;

  // Fill sparse matrix.
  sp_mat cleanedData = arma::sp_mat(locations, values, maxUserID, maxItemID);

  mlpack::math::RandomSeed(10);
  ValidationRMSETermination<sp_mat> vrt(cleanedData, 2000);
  AMF<ValidationRMSETermination<sp_mat>,
      RandomInitialization,
      SVDBatchLearning> amf_1(vrt,
                              RandomInitialization(),
                              SVDBatchLearning(0.0009, 0, 0, 0));

  mat m1,m2;
  double RMSE_1 = amf_1.Apply(cleanedData, 2, m1, m2);

  mlpack::math::RandomSeed(10);
  AMF<ValidationRMSETermination<sp_mat>,
      RandomInitialization,
      SVDBatchLearning> amf_2(vrt,
                              RandomInitialization(),
                              SVDBatchLearning(0.0009, 0.5, 0.5, 0.8));

  double RMSE_2 = amf_2.Apply(cleanedData, 2, m1, m2);

  BOOST_REQUIRE_LE(RMSE_2, RMSE_1);
}

/**
 * Make sure the SVD can factorize matrices with negative entries.
 */
BOOST_AUTO_TEST_CASE(SVDBatchNegativeElementTest)
{
  mat test;
  test.zeros(3,3);
  test(0, 0) = 1;
  test(0, 1) = -2;
  test(0, 2) = 3;
  test(1, 0) = 2;
  test(1, 1) = -1;
  test(1, 2) = 2;
  test(2, 0) = 2;
  test(2, 1) = 2;
  test(2, 2) = 2;

  AMF<SimpleToleranceTermination<mat>,
      RandomInitialization,
      SVDBatchLearning> amf(SimpleToleranceTermination<mat>(),
                            RandomInitialization(),
                            SVDBatchLearning(0.3, 0.001, 0.001, 0));
  mat m1, m2;
  amf.Apply(test, 2, m1, m2);

  arma::mat result = m1 * m2;

  for(size_t i = 0;i < 3;i++)
  {
    for(size_t j = 0;j < 3;j++)
    {
      BOOST_REQUIRE_LE(abs(test(i,j) - result(i,j)), 0.5);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
