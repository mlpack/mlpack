/**
 * @file softmax_regression_test.cpp
 * @author Siddharth Agrawal
 *
 * Test the SoftmaxRegression class.
 */
#define BOOST_TEST_MODULE MaximalInputs

#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_autoencoder/maximal_inputs.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(MaximalInputsTest);

BOOST_AUTO_TEST_CASE(ColumnToBlocksEvaluate)
{
  arma::mat w1(2, 4);
  w1<<0<<1<<2<<3<<arma::endr
    <<4<<5<<6<<7;

  arma::mat input(5,5);
  input.submat(0, 0, 1, 3) = w1;

  arma::mat maximalInputs;
  const std::pair<arma::uword, arma::uword> size =
          mlpack::nn::MaximalInputs(input, maximalInputs);

  arma::mat output;
  mlpack::nn::ColumnsToBlocks(maximalInputs, output, size.first, size.second);

  arma::mat matlabResults;
  matlabResults<<-1<<-1<<-1<<-1<<-1<<-1<<-1<<arma::endr
               <<-1<<-1<<-0.42857<<-1<<0.14286<<0.71429<<-1<<arma::endr
               <<-1<<-0.71429<<-0.14286<<-1.00000<<0.42857<<1<<-1<<arma::endr
               <<-1<<-1<<-1<<-1<<-1<<-1<<-1;

  BOOST_REQUIRE(matlabResults.n_rows == output.n_rows &&
                matlabResults.n_cols == output.n_cols);

  for(arma::uword i = 0; i != matlabResults.n_elem; ++i)
  {
    BOOST_REQUIRE_CLOSE(matlabResults[i], output[i], 1e-2);
  }
}

BOOST_AUTO_TEST_SUITE_END();
