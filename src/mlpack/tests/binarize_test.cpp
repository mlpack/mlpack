/**
 * @file binarize_test.cpp
 * @author Keon Kim
 *
 * Test the Binarzie method.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/binarize.hpp>
#include <mlpack/core/math/random.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::data;

BOOST_AUTO_TEST_SUITE(BinarizeTest);

/**
 * Compare the binarized data with answer.
 *
 * @param input The original data set before Binarize.
 * @param answer The data want to compare with the input.
 */
void CheckAnswer(const mat& input,
                 const umat& answer)
{
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    const mat& lhsCol = input.col(i);
    const umat& rhsCol = answer.col(i);
    for (size_t j = 0; j < lhsCol.n_rows; ++j)
    {
      if (std::abs(rhsCol(j)) < 1e-5)
        BOOST_REQUIRE_SMALL(lhsCol(j), 1e-5);
      else
        BOOST_REQUIRE_CLOSE(lhsCol(j), rhsCol(j), 1e-5);
    }
  }
}

BOOST_AUTO_TEST_CASE(BinarizeThreshold)
{
  mat input(10, 10, fill::randu); // fill input with randome Number
  mat constMat(10, 10);
  double threshold = math::Random(); // random number threshold
  constMat.fill(threshold);

  umat answer = input > constMat;

  // Binarize every values inside the matrix with threshold of 0;
  Binarize(input, threshold);

  CheckAnswer(input, answer);
}

BOOST_AUTO_TEST_SUITE_END();
