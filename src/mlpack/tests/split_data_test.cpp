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

void compareData(arma::mat const &inputData, arma::mat const &compareData,
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

BOOST_AUTO_TEST_CASE(SplitDataSplitResult)
{
    arma::mat trainData(2,10);
    trainData.randu();
    arma::Row<size_t> labels(trainData.n_cols);
    for(size_t i = 0; i != labels.n_cols; ++i){
        labels(i) = i;
    }

    auto const value = util::TrainTestSplit(trainData, labels, 0.2);
    BOOST_REQUIRE(std::get<0>(value).n_cols == 8);
    BOOST_REQUIRE(std::get<1>(value).n_cols == 2);
    BOOST_REQUIRE(std::get<2>(value).n_cols == 8);
    BOOST_REQUIRE(std::get<3>(value).n_cols == 2);

    compareData(trainData, std::get<0>(value), std::get<2>(value));
    compareData(trainData, std::get<1>(value), std::get<3>(value));
}

BOOST_AUTO_TEST_SUITE_END();
