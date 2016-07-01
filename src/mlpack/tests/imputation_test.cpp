/**
 * @file imputation_test.cpp
 * @author Keon Kim
 *
 * Tests for data::Imputer class
 */
#include <sstream>

#include <mlpack/core.hpp>
#include <mlpack/core/data/dataset_info.hpp>
#include <mlpack/core/data/map_policies/increment_policy.hpp>
#include <mlpack/core/data/map_policies/missing_policy.hpp>
#include <mlpack/core/data/imputer.hpp>
#include <mlpack/core/data/imputation_methods/custom_imputation.hpp>
#include <mlpack/core/data/imputation_methods/listwise_deletion.hpp>
#include <mlpack/core/data/imputation_methods/mean_imputation.hpp>
#include <mlpack/core/data/imputation_methods/median_imputation.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(ImputationTest);

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(DatasetMapperImputerTest)
{
  fstream f;
  f.open("test_file.csv", fstream::out);
  f << "a, 2, 3"  << endl;
  f << "5, 6, 7"  << endl;
  f << "8, 9, 10" << endl;
  f.close();

  arma::mat input;
  arma::mat output;
  string missingValue = "a";
  double customValue = 99;
  size_t feature = 0;

  DatasetInfo info;
  BOOST_REQUIRE(data::Load("test_file.csv", input, info) == true);

  BOOST_REQUIRE_EQUAL(input.n_rows, 3);
  BOOST_REQUIRE_EQUAL(input.n_cols, 3);

  /* TODO: Connect Load with the new DatasetMapper instead of DatasetInfo*/

  //Imputer<double,
          //DatasetInfo,
          //CustomImputation<double>> impu(info);
  //impu.Impute(input, output, missingValue, customValue, feature);
  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(CustomImputationTest)
{
  arma::mat input("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat outputT; // assume input is transposed
  arma::mat output;  // assume input is not transposed
  double customValue = 99;
  double mappedValue = 0.0;

  CustomImputation<double> imputer;

  // transposed
  imputer.Apply(input, outputT, mappedValue, customValue, 0/*dimension*/, true);

  BOOST_REQUIRE_CLOSE(outputT(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 1), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 3), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 3), 8.0, 1e-5);

  // not transposed
  imputer.Apply(input, output, mappedValue, customValue, 1/*dimension*/, false);

  BOOST_REQUIRE_CLOSE(output(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 1), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 3), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 3), 8.0, 1e-5);
}

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(MeanImputationTest)
{
  arma::mat input("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat outputT; // assume input is transposed
  arma::mat output;  // assume input is not transposed
  double mappedValue = 0.0;

  MeanImputation<double> imputer;

  // transposed
  imputer.Apply(input, outputT, mappedValue, 0/*dimension*/, true);

  BOOST_REQUIRE_CLOSE(outputT(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 1), 2.5, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 3), 2.5, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 3), 8.0, 1e-5);

  // not transposed
  imputer.Apply(input, output, mappedValue, 1/*dimension*/, false);

  BOOST_REQUIRE_CLOSE(output(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 1), 7.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 3), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 3), 8.0, 1e-5);
}

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(MedianImputationTest)
{
  arma::mat input("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat outputT; // assume input is transposed
  arma::mat output;  // assume input is not transposed
  double mappedValue = 0.0;

  MedianImputation<double> imputer;

  // transposed
  imputer.Apply(input, outputT, mappedValue, 1/*dimension*/, true);

  BOOST_REQUIRE_CLOSE(outputT(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 3), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 2), 5.5, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 3), 8.0, 1e-5);

  // not transposed
  imputer.Apply(input, output, mappedValue, 1/*dimension*/, false);

  BOOST_REQUIRE_CLOSE(output(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 3), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(2, 3), 8.0, 1e-5);
}

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(ListwiseDeletionTest)
{
  arma::mat input("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat outputT; // assume input is transposed
  arma::mat output;  // assume input is not transposed
  double mappedValue = 0.0;

  ListwiseDeletion<double> imputer;

  // transposed
  imputer.Apply(input, outputT, mappedValue, 0, true); // transposed

  BOOST_REQUIRE_CLOSE(outputT(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(0, 1), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(1, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(outputT(2, 1), 4.0, 1e-5);

  // not transposed
  imputer.Apply(input, output, mappedValue, 1, false); // not transposed

  BOOST_REQUIRE_CLOSE(output(0, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(0, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(output(1, 3), 8.0, 1e-5);
}


BOOST_AUTO_TEST_SUITE_END();
