/**
 * @file split_data_test.cpp
 * @author Tham Ngap Wei
 *
 * Test the SplitData method.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::data;

BOOST_AUTO_TEST_SUITE(SplitDataTest);

/**
 * Compare the data after train test split.  This assumes that the labels
 * correspond to each column, so that we can easily check each point against its
 * original.
 *
 * @param inputData The original data set before split.
 * @param compareData The data want to compare with the inputData;
 *   it could be train data or test data.
 * @param inputLabel The labels of each point in compareData.
 */
void CompareData(const mat& inputData,
                 const mat& compareData,
                 const Row<size_t>& inputLabel)
{
  for (size_t i = 0; i != compareData.n_cols; ++i)
  {
    const mat& lhsCol = inputData.col(inputLabel(i));
    const mat& rhsCol = compareData.col(i);
    for (size_t j = 0; j != lhsCol.n_rows; ++j)
    {
      BOOST_REQUIRE_CLOSE(lhsCol(j), rhsCol(j), 1e-5);
    }
  }
}

BOOST_AUTO_TEST_CASE(SplitDataSplitResultMat)
{
  mat input(2, 10);
  input.randu();

  // Set the labels to the column ID, so that CompareData can compare the data
  // after TrainTestSplit is called.
  const Row<size_t> labels = arma::linspace<Row<size_t>>(0, input.n_cols - 1,
      input.n_cols);

  auto const value = TrainTestSplit(input, labels, 0.2);
  BOOST_REQUIRE_EQUAL(std::get<0>(value).n_cols, 8);
  BOOST_REQUIRE_EQUAL(std::get<1>(value).n_cols, 2);
  BOOST_REQUIRE_EQUAL(std::get<2>(value).n_cols, 8);
  BOOST_REQUIRE_EQUAL(std::get<3>(value).n_cols, 2);

  CompareData(input, std::get<0>(value), std::get<2>(value));
  CompareData(input, std::get<1>(value), std::get<3>(value));
}

BOOST_AUTO_TEST_SUITE_END();
