#include <mlpack/core.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/update_rules/svd_incomplete_incremental_learning.hpp>
#include <mlpack/methods/amf/update_rules/svd_complete_incremental_learning.hpp>
#include <mlpack/methods/amf/init_rules/random_init.hpp>
#include <mlpack/methods/amf/termination_policies/incomplete_incremental_termination.hpp>
#include <mlpack/methods/amf/termination_policies/complete_incremental_termination.hpp>
#include <mlpack/methods/amf/termination_policies/simple_tolerance_termination.hpp>
#include <mlpack/methods/amf/termination_policies/validation_RMSE_termination.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(SVDIncrementalTest);

using namespace std;
using namespace mlpack;
using namespace mlpack::amf;
using namespace arma;

/**
 * Test for convergence of incomplete incremenal learning
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
BOOST_AUTO_TEST_CASE(SVDIncompleteIncrementalConvergenceTest)
{
  mlpack::math::RandomSeed(10);
  sp_mat data;
  data.sprandn(1000, 1000, 0.2);
  
  SVDIncompleteIncrementalLearning svd(0.01);
  IncompleteIncrementalTermination<SimpleToleranceTermination<sp_mat> > iit;
  
  AMF<IncompleteIncrementalTermination<SimpleToleranceTermination<sp_mat> >, 
      RandomInitialization, 
      SVDIncompleteIncrementalLearning> amf(iit, RandomInitialization(), svd);
  
  mat m1,m2;
  amf.Apply(data, 2, m1, m2);
  
  BOOST_REQUIRE_NE(amf.TerminationPolicy().Iteration(), 
                    amf.TerminationPolicy().MaxIterations());
}

/**
 * Test for convergence of complete incremenal learning
 */
BOOST_AUTO_TEST_CASE(SVDCompleteIncrementalConvergenceTest)
{
  mlpack::math::RandomSeed(10);
  sp_mat data;
  data.sprandn(1000, 1000, 0.2);
  
  SVDCompleteIncrementalLearning<sp_mat> svd(0.01);
  CompleteIncrementalTermination<SimpleToleranceTermination<sp_mat> > iit;
  
  AMF<CompleteIncrementalTermination<SimpleToleranceTermination<sp_mat> >, 
      RandomInitialization, 
      SVDCompleteIncrementalLearning<sp_mat> > amf(iit, 
                                                   RandomInitialization(), 
                                                   svd);
  mat m1,m2;
  amf.Apply(data, 2, m1, m2);
  
  BOOST_REQUIRE_NE(amf.TerminationPolicy().Iteration(), 
                    amf.TerminationPolicy().MaxIterations());
}


BOOST_AUTO_TEST_CASE(SVDIncompleteIncrementalRegularizationTest)
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
  sp_mat cleanedData2 = cleanedData;

  mlpack::math::RandomSeed(10);
  ValidationRMSETermination<sp_mat> vrt(cleanedData, 2000);
  AMF<IncompleteIncrementalTermination<ValidationRMSETermination<sp_mat> >,
      RandomInitialization,
      SVDIncompleteIncrementalLearning> amf_1(vrt,
                              RandomInitialization(),
                              SVDIncompleteIncrementalLearning(0.001, 0, 0));

  mat m1,m2;
  double RMSE_1 = amf_1.Apply(cleanedData, 2, m1, m2);

  mlpack::math::RandomSeed(10);
  ValidationRMSETermination<sp_mat> vrt2(cleanedData2, 2000);
  AMF<IncompleteIncrementalTermination<ValidationRMSETermination<sp_mat> >,
      RandomInitialization,
      SVDIncompleteIncrementalLearning> amf_2(vrt2,
                              RandomInitialization(),
                              SVDIncompleteIncrementalLearning(0.001, 0.01, 0.01));

  mat m3, m4;
  double RMSE_2 = amf_2.Apply(cleanedData2, 2, m3, m4);
  
  BOOST_REQUIRE_LT(RMSE_2, RMSE_1);
}

BOOST_AUTO_TEST_SUITE_END();
