/**
 * @file imputer_test.cpp
 * @author Keon Kim
 *
 * Tests for data::Imputer class
 */
#include <sstream>

#include <mlpack/core.hpp>
#include <mlpack/core/data/dataset_info.hpp>
#include <mlpack/core/data/map_policies/increment_policy.hpp>
#include <mlpack/core/data/imputer.hpp>
#include <mlpack/core/data/impute_strategies/custom_strategy.hpp>
#include <mlpack/core/data/impute_strategies/mean_strategy.hpp>
#include <mlpack/core/data/impute_strategies/median_strategy.hpp>
#include <mlpack/core/data/impute_strategies/mode_strategy.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(ImputerTest);

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(CustomStrategyTest)
{
  fstream f;
  f.open("test_file.csv", fstream::out);
  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;
  f.close();

  arma::mat test;
  using Mapper = DatasetMapper<IncrementPolicy>;
  Mapper info;
  BOOST_REQUIRE(data::Load("test_file.csv", test) == true);

  Imputer<arma::Mat<double>, Mapper, CustomStrategy> impu(info);
  impu.template Impute<double>(input,
                               output,
                               missingValue,
                               customValue,
                               feature);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(MeanStrategyTestt)
{
  fstream f;
  f.open("test_file.csv", fstream::out);
  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;
  f.close();

  arma::mat test;
  using Mapper = DatasetMapper<IncrementPolicy>;
  Mapper info;
  BOOST_REQUIRE(data::Load("test_file.csv", test) == true);

  Imputer<arma::Mat<double>, Mapper, CustomStrategy> impu(info);
  impu.template Impute<double>(input,
                               output,
                               missingValue,
                               customValue,
                               feature);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(MedianStrategyTestt)
{
  fstream f;
  f.open("test_file.csv", fstream::out);
  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;
  f.close();

  arma::mat test;
  using Mapper = DatasetMapper<IncrementPolicy>;
  Mapper info;
  BOOST_REQUIRE(data::Load("test_file.csv", test) == true);

  Imputer<arma::Mat<double>, Mapper, CustomStrategy> impu(info);
  impu.template Impute<double>(input,
                               output,
                               missingValue,
                               customValue,
                               feature);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(ModeStrategyTestt)
{
  fstream f;
  f.open("test_file.csv", fstream::out);
  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;
  f.close();

  arma::mat test;
  using Mapper = DatasetMapper<IncrementPolicy>;
  Mapper info;
  BOOST_REQUIRE(data::Load("test_file.csv", test) == true);

  Imputer<arma::Mat<double>, Mapper, CustomStrategy> impu(info);
  impu.template Impute<double>(input,
                               output,
                               missingValue,
                               customValue,
                               feature);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}


BOOST_AUTO_TEST_SUITE_END();
