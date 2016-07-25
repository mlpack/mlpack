/**
 * @file imputation_test.cpp
 * @author Keon Kim
 *
 * Tests for data::Imputer class
 */
#include <sstream>

#include <mlpack/core.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>
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
 * 1. Make sure a CSV is loaded correctly with mappings using MissingPolicy.
 * 2. Try Imputer object with CustomImputation method to impute data "a".
 * (It is ok to test on one method since the other ones will be covered in the
 * next cases).
 */
BOOST_AUTO_TEST_CASE(DatasetMapperImputerTest)
{
  fstream f;
  f.open("test_file.csv", fstream::out);
  f << "a, 2, 3"  << endl;
  f << "5, 6, a"  << endl;
  f << "8, 9, 10" << endl;
  f.close();

  arma::mat input;

  std::set<string> mset;
  mset.insert("a");
  MissingPolicy policy(mset);
  DatasetMapper<MissingPolicy> info(policy);
  BOOST_REQUIRE(data::Load("test_file.csv", input, info) == true);

  // row and column test.
  BOOST_REQUIRE_EQUAL(input.n_rows, 3);
  BOOST_REQUIRE_EQUAL(input.n_cols, 3);

  // Load check
  // MissingPolicy should convert strings to nans.
  BOOST_REQUIRE(std::isnan(input(0, 0)) == true);
  BOOST_REQUIRE_CLOSE(input(0, 1), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(0, 2), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(1, 0), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(1, 2), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(2, 0), 3.0, 1e-5);
  BOOST_REQUIRE(std::isnan(input(2, 1)) == true);
  BOOST_REQUIRE_CLOSE(input(2, 2), 10.0, 1e-5);

  // convert missing vals to 99.
  CustomImputation<double> customStrategy(99);
  Imputer<double,
          DatasetMapper<MissingPolicy>,
          CustomImputation<double>> imputer(info, customStrategy);
  // convert a or nan to 99 for dimension 0.
  imputer.Impute(input, "a", 0);

  // Custom imputation result check.
  BOOST_REQUIRE_CLOSE(input(0, 0), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(0, 1), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(0, 2), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(1, 0), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(1, 2), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(input(2, 0), 3.0, 1e-5);
  BOOST_REQUIRE(std::isnan(input(2, 1)) == true); // remains as NaN
  BOOST_REQUIRE_CLOSE(input(2, 2), 10.0, 1e-5);

  // Now, try impute() overload with all dimensions.
  arma::mat allInput;
  MissingPolicy allPolicy({"a"});
  DatasetMapper<MissingPolicy> allInfo(allPolicy);
  BOOST_REQUIRE(data::Load("test_file.csv", allInput, allInfo) == true);

  // convert missing vals to 99.
  CustomImputation<double> allCustomStrategy(99);
  Imputer<double,
          DatasetMapper<MissingPolicy>,
          CustomImputation<double>> allImputer(allInfo, allCustomStrategy);
  // convert a or nan to 99 for all dimensions.
  allImputer.Impute(allInput, "a");

  // Custom imputation result check
  BOOST_REQUIRE_CLOSE(allInput(0, 0), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(allInput(0, 1), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(allInput(0, 2), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(allInput(1, 0), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(allInput(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(allInput(1, 2), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(allInput(2, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(allInput(2, 1), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(allInput(2, 2), 10.0, 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure CustomImputation method replaces data 0 to 99.
 */
BOOST_AUTO_TEST_CASE(CustomImputationTest)
{
  arma::mat columnWiseInput("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat rowWiseInput(columnWiseInput);
  double customValue = 99;
  double mappedValue = 0.0;

  CustomImputation<double> imputer(customValue);

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 0/*dimension*/, true);

  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 1), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 3), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 3), 8.0, 1e-5);

  // row wise
  imputer.Impute(rowWiseInput, mappedValue, 1, false);

  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 1), 99.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 3), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 3), 8.0, 1e-5);
}

/**
 * Make sure MeanImputation method replaces data 0 to mean value of each
 * dimensions.
 */
BOOST_AUTO_TEST_CASE(MeanImputationTest)
{
  arma::mat columnWiseInput("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat rowWiseInput(columnWiseInput);
  double mappedValue = 0.0;

  MeanImputation<double> imputer;

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 0, true);

  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 1), 2.5, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 3), 2.5, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 3), 8.0, 1e-5);

  // row wise
  imputer.Impute(rowWiseInput, mappedValue, 1, false);

  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 1), 7.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 3), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 3), 8.0, 1e-5);
}

/**
 * Make sure MeanImputation method replaces data 0 to median value of each
 * dimensions.
 */
BOOST_AUTO_TEST_CASE(MedianImputationTest)
{
  arma::mat columnWiseInput("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat rowWiseInput(columnWiseInput);
  double mappedValue = 0.0;

  MedianImputation<double> imputer;

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 1, true);

  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 3), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 2), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 3), 8.0, 1e-5);

  // row wise
  imputer.Impute(rowWiseInput, mappedValue, 1, false);

  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 1), 7.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 2), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 3), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(2, 2), 4.0, 1e-5);
}

/**
 * Make sure ListwiseDeletion method deletes the whole column (if column wise)
 * or the row (if row wise) containing value of 0.
 */
BOOST_AUTO_TEST_CASE(ListwiseDeletionTest)
{
  arma::mat columnWiseInput("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat rowWiseInput(columnWiseInput);
  double mappedValue = 0.0;

  ListwiseDeletion<double> imputer;

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 0, true); // column wise

  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(0, 1), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(1, 1), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(columnWiseInput(2, 1), 4.0, 1e-5);

  // row wise
  imputer.Impute(rowWiseInput, mappedValue, 1, false); // row wise

  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 0), 5.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 1), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 2), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(0, 3), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 0), 9.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 1), 8.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 2), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(rowWiseInput(1, 3), 8.0, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
