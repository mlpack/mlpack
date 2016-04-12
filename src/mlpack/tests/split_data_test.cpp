/**
 * @file sparse_autoencoder_test.cpp
 * @author Siddharth Agrawal
 *
 * Test the SparseAutoencoder class.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/util/split_data.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;

BOOST_AUTO_TEST_SUITE(SplitDataTest);

/**
 * compare the data after train test split
 * @param inputData The original data set before split
 * @param compareData The data want to compare with the inputData,
 * it could be train data or test data
 * @param inputLabel The label of the compareData
 */
void CompareData(arma::mat const &inputData, arma::mat const &compareData,
                 arma::Row<size_t> const &inputLabel)
{
  for(size_t i = 0; i != compareData.n_cols; ++i){
    arma::mat const &lhsCol = inputData.col(inputLabel(i));
    arma::mat const &rhsCol = compareData.col(i);
    for(size_t j = 0; j != lhsCol.n_rows; ++j){
      BOOST_REQUIRE_CLOSE(lhsCol(j), rhsCol(j), 1e-5);
    }
  }
}

BOOST_AUTO_TEST_CASE(SplitDataSplitResultMat)
{    
  arma::mat input(2,10);
  input.randu();
  using Labels = arma::Row<size_t>;
  //set the labels range same as the col, so the CompareData
  //can compare the data after TrainTestSplit are valid or not
  Labels const labels =
          arma::linspace<Labels>(0, input.n_cols-1,
                                 input.n_cols);

  auto const value = util::TrainTestSplit(input, labels, 0.2);
  BOOST_REQUIRE(std::get<0>(value).n_cols == 8);
  BOOST_REQUIRE(std::get<1>(value).n_cols == 2);
  BOOST_REQUIRE(std::get<2>(value).n_cols == 8);
  BOOST_REQUIRE(std::get<3>(value).n_cols == 2);

  CompareData(input, std::get<0>(value), std::get<2>(value));
  CompareData(input, std::get<1>(value), std::get<3>(value));
}

BOOST_AUTO_TEST_SUITE_END();
