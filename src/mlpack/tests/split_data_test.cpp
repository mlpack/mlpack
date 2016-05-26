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

/**
 * Check that no labels have been duplicated.
 */
void CheckDuplication(const Row<size_t>& trainLabels,
                      const Row<size_t>& testLabels)
{
  // Assemble a vector that will hold the counts of each element.
  Row<size_t> counts(trainLabels.n_elem + testLabels.n_elem);
  counts.zeros();

  for (size_t i = 0; i < trainLabels.n_elem; ++i)
  {
    BOOST_REQUIRE_LT(trainLabels[i], counts.n_elem);
    counts[trainLabels[i]]++;
  }
  for (size_t i = 0; i < testLabels.n_elem; ++i)
  {
    BOOST_REQUIRE_LT(testLabels[i], counts.n_elem);
    counts[testLabels[i]]++;
  }

  // Now make sure each point has been used once.
  for (size_t i = 0; i < counts.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(counts[i], 1);
}

BOOST_AUTO_TEST_CASE(SplitDataSplitResultMat)
{
  mat input(2, 10);
  input.randu();

  // Set the labels to the column ID, so that CompareData can compare the data
  // after Split is called.
  const Row<size_t> labels = arma::linspace<Row<size_t>>(0, input.n_cols - 1,
      input.n_cols);

  const auto value = Split(input, labels, 0.2);
  BOOST_REQUIRE_EQUAL(std::get<0>(value).n_cols, 8);
  BOOST_REQUIRE_EQUAL(std::get<1>(value).n_cols, 2);
  BOOST_REQUIRE_EQUAL(std::get<2>(value).n_cols, 8);
  BOOST_REQUIRE_EQUAL(std::get<3>(value).n_cols, 2);

  CompareData(input, std::get<0>(value), std::get<2>(value));
  CompareData(input, std::get<1>(value), std::get<3>(value));

  // The last thing to check is that we aren't duplicating any points in the
  // train or test labels.
  CheckDuplication(std::get<2>(value), std::get<3>(value));
}

/**
 * The same test as above, but on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(SplitDataLargerTest)
{
  mat input(10, 497);
  input.randu();

  // Set the labels to the column ID.
  const Row<size_t> labels = arma::linspace<Row<size_t>>(0, input.n_cols - 1,
      input.n_cols);

  const auto value = Split(input, labels, 0.3);
  BOOST_REQUIRE_EQUAL(std::get<0>(value).n_cols, 497 - size_t(0.3 * 497));
  BOOST_REQUIRE_EQUAL(std::get<1>(value).n_cols, size_t(0.3 * 497));
  BOOST_REQUIRE_EQUAL(std::get<2>(value).n_cols, 497 - size_t(0.3 * 497));
  BOOST_REQUIRE_EQUAL(std::get<3>(value).n_cols, size_t(0.3 * 497));

  CompareData(input, std::get<0>(value), std::get<2>(value));
  CompareData(input, std::get<1>(value), std::get<3>(value));

  CheckDuplication(std::get<2>(value), std::get<3>(value));
}

BOOST_AUTO_TEST_SUITE_END();
